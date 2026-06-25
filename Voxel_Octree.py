import os
import csv
import gc
import time

import numpy as np
import psutil
import trimesh

# ============================================================
# CONFIGURAÇÕES
# ============================================================

MODELOS = {
    "Armadillo": "data/Armadillo.ply",
    "Buda": "data/Buda.ply",
}

Ns = [25, 50, 100, 200, 400]
TIPOS_GRADE = ["Justa", "Cubica"]
REPETICOES_OCTREE = 10

PASTA_RESULTADOS = "resultados"
ARQUIVO_CSV = os.path.join(PASTA_RESULTADOS, "resultados_octree_python.csv")

# Exporta os pontos ocupados em TXT uma única vez por modelo/N/tipo de grade.
EXPORTAR_POINTS_TXT = True

os.makedirs(PASTA_RESULTADOS, exist_ok=True)

process = psutil.Process(os.getpid())


# ============================================================
# OCTREE
# ============================================================

class OctreeNode:
    __slots__ = (
        "x0", "y0", "z0",
        "size_x", "size_y", "size_z",
        "is_leaf", "is_full", "children"
    )

    def __init__(self, x0, y0, z0, sx, sy, sz):
        self.x0, self.y0, self.z0 = x0, y0, z0
        self.size_x, self.size_y, self.size_z = sx, sy, sz
        self.is_leaf = True
        self.is_full = False
        self.children = []


def build_octree(x0, y0, z0, sx, sy, sz, points):
    """Constrói a octree usando a mesma lógica dos códigos originais."""
    node = OctreeNode(x0, y0, z0, sx, sy, sz)
    total_cells = sx * sy * sz

    if len(points) == 0:
        return node

    if sx <= 1 and sy <= 1 and sz <= 1:
        node.is_full = True
        return node

    if len(points) == total_cells:
        node.is_full = True
        return node

    node.is_leaf = False

    mid_x = x0 + sx // 2
    mid_y = y0 + sy // 2
    mid_z = z0 + sz // 2

    sx1, sx2 = sx // 2, sx - sx // 2
    sy1, sy2 = sy // 2, sy - sy // 2
    sz1, sz2 = sz // 2, sz - sz // 2

    child_defs = [
        (x0, y0, z0, sx1, sy1, sz1),
        (x0, y0, mid_z, sx1, sy1, sz2),
        (x0, mid_y, z0, sx1, sy2, sz1),
        (x0, mid_y, mid_z, sx1, sy2, sz2),
        (mid_x, y0, z0, sx2, sy1, sz1),
        (mid_x, y0, mid_z, sx2, sy1, sz2),
        (mid_x, mid_y, z0, sx2, sy2, sz1),
        (mid_x, mid_y, mid_z, sx2, sy2, sz2),
    ]

    buckets = [[] for _ in range(8)]

    for p in points:
        i, j, k = p
        bit_x = 1 if i >= mid_x else 0
        bit_y = 1 if j >= mid_y else 0
        bit_z = 1 if k >= mid_z else 0
        child_idx = (bit_x << 2) | (bit_y << 1) | bit_z
        buckets[child_idx].append(p)

    for idx, child in enumerate(child_defs):
        cx, cy, cz, csx, csy, csz = child
        node.children.append(build_octree(cx, cy, cz, csx, csy, csz, buckets[idx]))

    return node


def collect_stats(node, depth=0):
    total = 1
    leaves_full = 0
    leaves_empty = 0
    max_depth = depth

    if node.is_leaf:
        if node.is_full:
            leaves_full = 1
        else:
            leaves_empty = 1
    else:
        for child in node.children:
            t, f, e, d = collect_stats(child, depth + 1)
            total += t
            leaves_full += f
            leaves_empty += e
            max_depth = max(max_depth, d)

    return total, leaves_full, leaves_empty, max_depth


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def fmt(x, dec=4):
    """Formata número com vírgula decimal para o CSVs."""
    return f"{x:.{dec}f}".replace(".", ",")


def slug(texto):
    return texto.lower().replace("ú", "u").replace("á", "a").replace("í", "i")


def carregar_malha(caminho_modelo):
    mesh = trimesh.load(caminho_modelo)

    # Caso o arquivo seja carregado como Scene, junta as geometrias em uma única malha.
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    return mesh


def export_points_txt(points, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for x, y, z in points:
            f.write(f"{x} {y} {z}\n")


def preparar_grade(tipo_grade, vox, N):
    """
    Retorna nx, ny, nz e pontos ocupados conforme o tipo de grade.

    Justa:
        Usa diretamente as dimensões geradas pela voxelização do trimesh.

    Cubica:
        Usa uma grade N x N x N e mantém apenas os pontos dentro desse limite,
        seguindo a lógica do código cúbico original.
    """
    raw_points = np.array(vox.sparse_indices, dtype=np.int32)

    if tipo_grade == "Justa":
        nx, ny, nz = vox.shape
        points = [tuple(p) for p in raw_points]
        return nx, ny, nz, points

    if tipo_grade == "Cubica":
        nx, ny, nz = N, N, N
        points = []

        for p in raw_points:
            x, y, z = int(p[0]), int(p[1]), int(p[2])

            if 0 <= x < N and 0 <= y < N and 0 <= z < N:
                points.append((x, y, z))

        return nx, ny, nz, points

    raise ValueError(f"Tipo de grade inválido: {tipo_grade}")


def escrever_cabecalho_csv(writer):
    writer.writerow([
        "modelo",
        "tipo_grade",
        "N",
        "execucao",
        "voxel_size",
        "nx",
        "ny",
        "nz",
        "total",
        "occupied",
        "empty",
        "density",
        "t_vox",
        "delta_rss_vox",
        "octree_nodes",
        "leaves_full",
        "leaves_empty",
        "max_depth",
        "t_oct",
        "delta_rss_oct",
        "compression",
    ])


def escrever_linha_csv(
    writer,
    modelo,
    tipo_grade,
    N,
    execucao,
    voxel_size,
    nx,
    ny,
    nz,
    total_voxels,
    occupied,
    empty,
    density,
    t_vox,
    delta_rss_vox,
    total_nodes,
    leaves_full,
    leaves_empty,
    max_depth,
    t_oct,
    delta_rss_oct,
    compression,
):
    writer.writerow([
        modelo,
        tipo_grade,
        N,
        execucao,
        fmt(voxel_size),
        nx,
        ny,
        nz,
        total_voxels,
        occupied,
        empty,
        fmt(density),
        fmt(t_vox, 3),
        fmt(delta_rss_vox, 2),
        total_nodes,
        leaves_full,
        leaves_empty,
        max_depth,
        fmt(t_oct, 3),
        fmt(delta_rss_oct, 2),
        fmt(compression),
    ])


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main():
    with open(ARQUIVO_CSV, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv, delimiter=";")
        escrever_cabecalho_csv(writer)

        for nome_modelo, caminho_modelo in MODELOS.items():
            if not os.path.exists(caminho_modelo):
                print(f"[AVISO] Modelo não encontrado: {caminho_modelo}")
                continue

            print(f"\n=== Modelo: {nome_modelo} ===")
            mesh = carregar_malha(caminho_modelo)

            bounds = mesh.bounds
            extents = bounds[1] - bounds[0]
            max_extent = max(extents)

            for N in Ns:
                voxel_size = max_extent / N

                print(f"\nVoxelizando {nome_modelo} | N={N} | voxel_size={voxel_size:.6f}")

                mem_antes_vox = process.memory_info().rss
                t0_vox = time.perf_counter()
                vox = mesh.voxelized(pitch=voxel_size)
                t1_vox = time.perf_counter()
                mem_depois_vox = process.memory_info().rss

                t_vox = t1_vox - t0_vox
                delta_rss_vox = (mem_depois_vox - mem_antes_vox) / 1024**2

                for tipo_grade in TIPOS_GRADE:
                    nx, ny, nz, points = preparar_grade(tipo_grade, vox, N)

                    total_voxels = nx * ny * nz
                    occupied = len(points)
                    empty = total_voxels - occupied
                    density = occupied / total_voxels if total_voxels > 0 else 0

                    if EXPORTAR_POINTS_TXT:
                        nome_txt = f"{slug(nome_modelo)}_{slug(tipo_grade)}_points_{N}.txt"
                        caminho_txt = os.path.join(PASTA_RESULTADOS, nome_txt)
                        export_points_txt(points, caminho_txt)

                    print(
                        f"Grade {tipo_grade} | dimensões={nx}x{ny}x{nz} | "
                        f"ocupados={occupied} | execuções octree={REPETICOES_OCTREE}"
                    )

                    for execucao in range(1, REPETICOES_OCTREE + 1):
                        gc.collect()

                        mem_antes_oct = process.memory_info().rss
                        t0_oct = time.perf_counter()
                        root = build_octree(0, 0, 0, nx, ny, nz, points)
                        t1_oct = time.perf_counter()
                        mem_depois_oct = process.memory_info().rss

                        t_oct = t1_oct - t0_oct
                        delta_rss_oct = (mem_depois_oct - mem_antes_oct) / 1024**2

                        total_nodes, leaves_full, leaves_empty, max_depth = collect_stats(root)
                        compression = total_nodes / total_voxels if total_voxels > 0 else 0

                        escrever_linha_csv(
                            writer,
                            nome_modelo,
                            tipo_grade,
                            N,
                            execucao,
                            voxel_size,
                            nx,
                            ny,
                            nz,
                            total_voxels,
                            occupied,
                            empty,
                            density,
                            t_vox,
                            delta_rss_vox,
                            total_nodes,
                            leaves_full,
                            leaves_empty,
                            max_depth,
                            t_oct,
                            delta_rss_oct,
                            compression,
                        )
                        f_csv.flush()

                        print(
                            f"  Execução {execucao:02d}/{REPETICOES_OCTREE} | "
                            f"t_oct={t_oct:.4f}s | nós={total_nodes} | "
                            f"RSS_oct={delta_rss_oct:.2f} MB"
                        )

                        del root
                        gc.collect()

                    del points
                    gc.collect()

                del vox
                gc.collect()

            del mesh
            gc.collect()

    print(f"\nConcluído. CSV salvo em: {ARQUIVO_CSV}")


if __name__ == "__main__":
    main()
