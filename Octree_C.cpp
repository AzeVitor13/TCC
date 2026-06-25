#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <array>
#include <numeric>
#include <map>
#include <set>
#include <cctype>
#include <direct.h>

#include <windows.h>
#include <psapi.h>

// ============================================================
// CONFIGURACOES
// ============================================================
//
// Este codigo le os arquivos TXT exportados pelo Python combinado:
//
// resultados/armadillo_justa_points_25.txt
// resultados/armadillo_cubica_points_25.txt
// resultados/buda_justa_points_25.txt
// resultados/buda_cubica_points_25.txt
//
// Para recuperar voxel_size, nx, ny e nz com precisao, ele tambem tenta ler:
//
// resultados/resultados_octree_python_10_execucoes.csv
//
// A octree e reconstruida 10 vezes para cada combinacao:
// modelo x tipo de grade x N.
// ============================================================

const std::string PASTA_RESULTADOS = "resultados";
const std::string CSV_PYTHON = PASTA_RESULTADOS + "/resultados_octree_python.csv";
const std::string CSV_SAIDA = PASTA_RESULTADOS + "/resultados_octree_cpp.csv";

const int REPETICOES_OCTREE = 10;

std::vector<std::string> MODELOS = {
    "Armadillo",
    "Buda"
};

std::vector<std::string> TIPOS_GRADE = {
    "Justa",
    "Cubica"
};

std::vector<int> Ns = {
    25, 50, 100, 200, 400
};

// ============================================================
// ESTRUTURAS
// ============================================================

struct Point3D {
    int x, y, z;
};

struct GridSpec {
    std::string modelo;
    std::string tipo_grade;
    int N = 0;
    double voxel_size = 0.0;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    bool loaded_from_csv = false;
};

struct OctreeNode {
    int x0, y0, z0;
    int size_x, size_y, size_z;
    bool is_leaf;
    bool is_full;
    std::vector<std::unique_ptr<OctreeNode>> children;

    OctreeNode(int x, int y, int z, int sx, int sy, int sz)
        : x0(x), y0(y), z0(z),
          size_x(sx), size_y(sy), size_z(sz),
          is_leaf(true), is_full(false) {}
};

struct OctreeStats {
    long long total_nodes = 0;
    long long leaves_full = 0;
    long long leaves_empty = 0;
    int max_depth = 0;
};

struct ResultRow {
    std::string modelo;
    std::string tipo_grade;
    int N = 0;
    int execucao = 0;
    double voxel_size = 0.0;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    long long total_voxels = 0;
    long long occupied = 0;
    long long empty = 0;
    double density = 0.0;
    long long octree_nodes = 0;
    long long leaves_full = 0;
    long long leaves_empty = 0;
    int max_depth = 0;
    double t_oct = 0.0;
    double delta_rss_oct = 0.0;
    double compression = 0.0;
};

// ============================================================
// FUNCOES AUXILIARES
// ============================================================

long long get_rss_memory() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<long long>(pmc.WorkingSetSize);
    }
    return 0;
}

std::string to_lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

std::string slug(const std::string& s) {
    std::string out = to_lower_ascii(s);

    if (out == "cubica") return "cubica";
    if (out == "cubic") return "cubica";
    if (out == "justa") return "justa";
    if (out == "tight") return "justa";
    if (out == "armadillo") return "armadillo";
    if (out == "buda") return "buda";

    return out;
}

std::string make_key(const std::string& modelo, const std::string& tipo_grade, int N) {
    return slug(modelo) + "|" + slug(tipo_grade) + "|" + std::to_string(N);
}

std::vector<std::string> split_semicolon(const std::string& line) {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string item;

    while (std::getline(ss, item, ';')) {
        tokens.push_back(item);
    }

    return tokens;
}

double parse_double_br(const std::string& s) {
    if (s.empty()) return 0.0;

    std::string temp = s;
    std::replace(temp.begin(), temp.end(), ',', '.');
    return std::stod(temp);
}

std::string format_double_br(double value, int precision) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(precision) << value;

    std::string s = ss.str();
    std::replace(s.begin(), s.end(), '.', ',');

    return s;
}

int find_col(const std::vector<std::string>& header, const std::string& name) {
    for (int i = 0; i < static_cast<int>(header.size()); ++i) {
        if (header[i] == name) return i;
    }
    return -1;
}

std::map<std::string, GridSpec> load_grid_specs_from_combined_python_csv(const std::string& filename) {
    std::map<std::string, GridSpec> specs;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "[AVISO] CSV do Python nao encontrado: " << filename << "\n";
        std::cerr << "        O programa tentara inferir nx, ny e nz pelos TXT.\n";
        return specs;
    }

    std::string line;
    if (!std::getline(file, line)) {
        return specs;
    }

    std::vector<std::string> header = split_semicolon(line);

    int col_modelo = find_col(header, "modelo");
    int col_grade = find_col(header, "tipo_grade");
    int col_N = find_col(header, "N");
    int col_voxel = find_col(header, "voxel_size");
    int col_nx = find_col(header, "nx");
    int col_ny = find_col(header, "ny");
    int col_nz = find_col(header, "nz");

    if (col_modelo < 0 || col_grade < 0 || col_N < 0 ||
        col_voxel < 0 || col_nx < 0 || col_ny < 0 || col_nz < 0) {
        std::cerr << "[AVISO] Cabecalho inesperado no CSV do Python: " << filename << "\n";
        std::cerr << "        Esperado: modelo;tipo_grade;N;...;voxel_size;nx;ny;nz;...\n";
        return specs;
    }

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::vector<std::string> cols = split_semicolon(line);
        int max_col = std::max({col_modelo, col_grade, col_N, col_voxel, col_nx, col_ny, col_nz});
        if (static_cast<int>(cols.size()) <= max_col) continue;

        GridSpec spec;
        spec.modelo = cols[col_modelo];
        spec.tipo_grade = cols[col_grade];
        spec.N = std::stoi(cols[col_N]);
        spec.voxel_size = parse_double_br(cols[col_voxel]);
        spec.nx = std::stoi(cols[col_nx]);
        spec.ny = std::stoi(cols[col_ny]);
        spec.nz = std::stoi(cols[col_nz]);
        spec.loaded_from_csv = true;

        // O CSV do Python tem 10 linhas por combinacao. Guardamos apenas a primeira,
        // pois as dimensoes e o voxel_size sao os mesmos para as 10 execucoes.
        std::string key = make_key(spec.modelo, spec.tipo_grade, spec.N);
        if (specs.find(key) == specs.end()) {
            specs[key] = spec;
        }
    }

    std::cout << "Specs carregadas do CSV do Python: " << specs.size() << " combinacoes.\n";
    return specs;
}

bool load_points_txt(const std::string& filename, std::vector<Point3D>& points) {
    points.clear();
    std::ifstream file(filename);

    if (!file.is_open()) {
        return false;
    }

    Point3D p;
    while (file >> p.x >> p.y >> p.z) {
        points.push_back(p);
    }

    return true;
}

GridSpec infer_grid_spec_from_points(
    const std::string& modelo,
    const std::string& tipo_grade,
    int N,
    const std::vector<Point3D>& points
) {
    GridSpec spec;
    spec.modelo = modelo;
    spec.tipo_grade = tipo_grade;
    spec.N = N;
    spec.voxel_size = 0.0;
    spec.loaded_from_csv = false;

    if (slug(tipo_grade) == "cubica") {
        spec.nx = N;
        spec.ny = N;
        spec.nz = N;
        return spec;
    }

    int max_x = 0;
    int max_y = 0;
    int max_z = 0;

    for (const Point3D& p : points) {
        max_x = std::max(max_x, p.x);
        max_y = std::max(max_y, p.y);
        max_z = std::max(max_z, p.z);
    }

    spec.nx = max_x + 1;
    spec.ny = max_y + 1;
    spec.nz = max_z + 1;

    return spec;
}

// ============================================================
// OCTREE
// ============================================================

std::unique_ptr<OctreeNode> build_octree(
    int x0, int y0, int z0,
    int sx, int sy, int sz,
    const std::vector<Point3D>& points,
    const std::vector<int>& indices
) {
    auto node = std::make_unique<OctreeNode>(x0, y0, z0, sx, sy, sz);

    long long total_cells =
        static_cast<long long>(sx) *
        static_cast<long long>(sy) *
        static_cast<long long>(sz);

    if (indices.empty()) {
        node->is_full = false;
        return node;
    }

    if (sx <= 1 && sy <= 1 && sz <= 1) {
        node->is_full = true;
        return node;
    }

    if (static_cast<long long>(indices.size()) == total_cells) {
        node->is_full = true;
        return node;
    }

    node->is_leaf = false;
    node->children.reserve(8);

    int mid_x = x0 + sx / 2;
    int mid_y = y0 + sy / 2;
    int mid_z = z0 + sz / 2;

    int sx1 = sx / 2;
    int sy1 = sy / 2;
    int sz1 = sz / 2;

    int sx2 = sx - sx1;
    int sy2 = sy - sy1;
    int sz2 = sz - sz1;

    struct ChildDef {
        int cx, cy, cz;
        int csx, csy, csz;
    };

    std::array<ChildDef, 8> defs = {{
        {x0,    y0,    z0,    sx1, sy1, sz1},
        {x0,    y0,    mid_z, sx1, sy1, sz2},
        {x0,    mid_y, z0,    sx1, sy2, sz1},
        {x0,    mid_y, mid_z, sx1, sy2, sz2},
        {mid_x, y0,    z0,    sx2, sy1, sz1},
        {mid_x, y0,    mid_z, sx2, sy1, sz2},
        {mid_x, mid_y, z0,    sx2, sy2, sz1},
        {mid_x, mid_y, mid_z, sx2, sy2, sz2}
    }};

    std::array<std::vector<int>, 8> buckets;

    for (auto& bucket : buckets) {
        bucket.reserve(indices.size() / 8);
    }

    for (int idx : indices) {
        const Point3D& p = points[idx];

        int bit_x = (p.x >= mid_x) ? 1 : 0;
        int bit_y = (p.y >= mid_y) ? 1 : 0;
        int bit_z = (p.z >= mid_z) ? 1 : 0;

        int child_idx = (bit_x << 2) | (bit_y << 1) | bit_z;
        buckets[child_idx].push_back(idx);
    }

    for (int i = 0; i < 8; ++i) {
        node->children.push_back(
            build_octree(
                defs[i].cx, defs[i].cy, defs[i].cz,
                defs[i].csx, defs[i].csy, defs[i].csz,
                points,
                buckets[i]
            )
        );
    }

    return node;
}

OctreeStats collect_stats(const OctreeNode* node, int depth = 0) {
    OctreeStats stats;

    stats.total_nodes = 1;
    stats.max_depth = depth;

    if (node->is_leaf) {
        if (node->is_full) {
            stats.leaves_full = 1;
        } else {
            stats.leaves_empty = 1;
        }
        return stats;
    }

    for (const auto& child : node->children) {
        OctreeStats child_stats = collect_stats(child.get(), depth + 1);

        stats.total_nodes += child_stats.total_nodes;
        stats.leaves_full += child_stats.leaves_full;
        stats.leaves_empty += child_stats.leaves_empty;
        stats.max_depth = std::max(stats.max_depth, child_stats.max_depth);
    }

    return stats;
}

// ============================================================
// CSV
// ============================================================

void save_results_csv(const std::string& filename, const std::vector<ResultRow>& results) {
    std::ofstream f(filename);

    if (!f.is_open()) {
        std::cerr << "Erro ao criar CSV: " << filename << "\n";
        return;
    }

    f << "modelo;tipo_grade;N;execucao;voxel_size;nx;ny;nz;total;occupied;empty;density;"
      << "octree_nodes;leaves_full;leaves_empty;max_depth;"
      << "t_oct;delta_rss_oct;compression\n";

    for (const auto& r : results) {
        f << r.modelo << ";"
          << r.tipo_grade << ";"
          << r.N << ";"
          << r.execucao << ";"
          << format_double_br(r.voxel_size, 4) << ";"
          << r.nx << ";"
          << r.ny << ";"
          << r.nz << ";"
          << r.total_voxels << ";"
          << r.occupied << ";"
          << r.empty << ";"
          << format_double_br(r.density, 4) << ";"
          << r.octree_nodes << ";"
          << r.leaves_full << ";"
          << r.leaves_empty << ";"
          << r.max_depth << ";"
          << format_double_br(r.t_oct, 5) << ";"
          << format_double_br(r.delta_rss_oct, 2) << ";"
          << format_double_br(r.compression, 4) << "\n";
    }
}

void append_result_row(
    std::vector<ResultRow>& results,
    const GridSpec& spec,
    int execucao,
    long long total_voxels,
    long long occupied,
    long long empty,
    double density,
    const OctreeStats& stats,
    double t_oct,
    double delta_rss_oct,
    double compression
) {
    ResultRow row;
    row.modelo = spec.modelo;
    row.tipo_grade = spec.tipo_grade;
    row.N = spec.N;
    row.execucao = execucao;
    row.voxel_size = spec.voxel_size;
    row.nx = spec.nx;
    row.ny = spec.ny;
    row.nz = spec.nz;
    row.total_voxels = total_voxels;
    row.occupied = occupied;
    row.empty = empty;
    row.density = density;
    row.octree_nodes = stats.total_nodes;
    row.leaves_full = stats.leaves_full;
    row.leaves_empty = stats.leaves_empty;
    row.max_depth = stats.max_depth;
    row.t_oct = t_oct;
    row.delta_rss_oct = delta_rss_oct;
    row.compression = compression;

    results.push_back(row);
}

// ============================================================
// EXECUCAO DO EXPERIMENTO
// ============================================================

void process_experiment(
    const std::string& modelo,
    const std::string& tipo_grade,
    int N,
    const std::map<std::string, GridSpec>& specs_csv,
    std::vector<ResultRow>& results
) {
    std::string modelo_slug = slug(modelo);
    std::string grade_slug = slug(tipo_grade);

    std::string points_file =
        PASTA_RESULTADOS + "/" + modelo_slug + "_" + grade_slug + "_points_" + std::to_string(N) + ".txt";

    std::vector<Point3D> points;
    if (!load_points_txt(points_file, points)) {
        std::cerr << "[AVISO] Arquivo de pontos nao encontrado: " << points_file << "\n";
        return;
    }

    if (points.empty()) {
        std::cerr << "[AVISO] Arquivo de pontos vazio: " << points_file << "\n";
        return;
    }

    GridSpec spec;
    std::string key = make_key(modelo, tipo_grade, N);
    auto it = specs_csv.find(key);

    if (it != specs_csv.end()) {
        spec = it->second;
    } else {
        spec = infer_grid_spec_from_points(modelo, tipo_grade, N, points);
        std::cerr << "[AVISO] Spec nao encontrada no CSV para "
                  << modelo << " | " << tipo_grade << " | N=" << N << ".\n";
        std::cerr << "        Dimensoes inferidas pelos TXT. voxel_size sera 0,0000 no CSV.\n";
    }

    if (spec.modelo.empty()) spec.modelo = modelo;
    if (spec.tipo_grade.empty()) spec.tipo_grade = tipo_grade;

    long long total_voxels =
        static_cast<long long>(spec.nx) *
        static_cast<long long>(spec.ny) *
        static_cast<long long>(spec.nz);

    long long occupied = static_cast<long long>(points.size());
    long long empty = total_voxels - occupied;

    double density =
        total_voxels > 0
        ? static_cast<double>(occupied) / static_cast<double>(total_voxels)
        : 0.0;

    std::vector<int> indices(points.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::cout << "\nModelo: " << spec.modelo
              << " | Grade: " << spec.tipo_grade
              << " | N=" << spec.N
              << " | dimensoes=" << spec.nx << "x" << spec.ny << "x" << spec.nz
              << " | ocupados=" << occupied << "\n";

    for (int execucao = 1; execucao <= REPETICOES_OCTREE; ++execucao) {
        long long mem_before = get_rss_memory();
        auto t0 = std::chrono::high_resolution_clock::now();

        std::unique_ptr<OctreeNode> root =
            build_octree(
                0, 0, 0,
                spec.nx, spec.ny, spec.nz,
                points,
                indices
            );

        auto t1 = std::chrono::high_resolution_clock::now();
        long long mem_after = get_rss_memory();

        double t_oct = std::chrono::duration<double>(t1 - t0).count();
        double delta_rss_oct =
            static_cast<double>(mem_after - mem_before) / (1024.0 * 1024.0);

        OctreeStats stats = collect_stats(root.get());

        double compression =
            total_voxels > 0
            ? static_cast<double>(stats.total_nodes) / static_cast<double>(total_voxels)
            : 0.0;

        append_result_row(
            results,
            spec,
            execucao,
            total_voxels,
            occupied,
            empty,
            density,
            stats,
            t_oct,
            delta_rss_oct,
            compression
        );

        std::cout << "  Execucao " << std::setw(2) << std::setfill('0') << execucao
                  << std::setfill(' ')
                  << "/" << REPETICOES_OCTREE
                  << " | t_oct=" << std::fixed << std::setprecision(5) << t_oct << "s"
                  << " | nos=" << stats.total_nodes
                  << " | RSS_oct=" << std::fixed << std::setprecision(2) << delta_rss_oct << " MB\n";

        root.reset();
    }
}

int main() {
    _mkdir(PASTA_RESULTADOS.c_str());

    std::map<std::string, GridSpec> specs_csv =
        load_grid_specs_from_combined_python_csv(CSV_PYTHON);

    std::vector<ResultRow> results;
    results.reserve(MODELOS.size() * TIPOS_GRADE.size() * Ns.size() * REPETICOES_OCTREE);

    for (const std::string& modelo : MODELOS) {
        for (const std::string& tipo_grade : TIPOS_GRADE) {
            for (int N : Ns) {
                process_experiment(modelo, tipo_grade, N, specs_csv, results);
            }
        }
    }

    save_results_csv(CSV_SAIDA, results);

    std::cout << "\nConcluido. CSV salvo em: " << CSV_SAIDA << "\n";
    std::cout << "Total de linhas exportadas: " << results.size() << "\n";

    return 0;
}
