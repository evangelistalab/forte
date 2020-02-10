#include <iostream>
#include <fstream> // std::filebuf

#include "hayai/hayai.hpp"
#include "hayai/hayai_main.hpp"

#include "src/sparse_ci/determinant.h"

using namespace forte;


int main(int argc, char* argv[]) {
    std::filebuf fb;
    fb.open("bench.json", std::ios::out);
    std::ostream json(&fb);

    // Set up the main runner.
    hayai::MainRunner runner;

    // Parse the arguments.
    int result = runner.ParseArgs(argc, argv);
    if (result)
        return result;

    //    hayai::ConsoleOutputter consoleOutputter;
    //    hayai::JsonOutputter JSONOutputter(json);

    //    hayai::Benchmarker::AddOutputter(consoleOutputter);
    //    hayai::Benchmarker::AddOutputter(JSONOutputter);
    //    hayai::Benchmarker::RunAllTests();

    // Execute based on the selected mode.
    return runner.Run();
    fb.close();
    return 0;
}

Determinant make_det_from_string(std::string s_a, std::string s_b) {
    Determinant d;
    if (s_a.size() == s_b.size()) {
        for (std::string::size_type i = 0; i < s_a.size(); ++i) {
            d.set_alfa_bit(i, s_a[i] == '0' ? 0 : 1);
            d.set_beta_bit(i, s_b[i] == '0' ? 0 : 1);
        }
    } else {
        std::cout << "\n\n  Function make_det_from_string called with strings of different size";
        exit(1);
    }
    return d;
}

Determinant det_test =
    make_det_from_string("1001100000000000000000000000000000000000000000000000000000010000",
                         "0001000000000000001000000000000000000000000000000000000000000001");

BENCHMARK(Determinant, count, 10, 100000) {
    det_test.count_alfa();
    det_test.count_beta();
}

BENCHMARK_P(Determinant, sign_a, 10, 100000, (std::size_t n)) { det_test.slater_sign_a(n); }

BENCHMARK_P_INSTANCE(Determinant, sign_a, (2));
BENCHMARK_P_INSTANCE(Determinant, sign_a, (16));
BENCHMARK_P_INSTANCE(Determinant, sign_a, (32));
BENCHMARK_P_INSTANCE(Determinant, sign_a, (63));

BENCHMARK_P(Determinant, sign_aa, 10, 100000, (std::size_t m, std::size_t n)) {
    det_test.slater_sign_aa(m, n);
}

BENCHMARK_P_INSTANCE(Determinant, sign_aa, (1, 2));
BENCHMARK_P_INSTANCE(Determinant, sign_aa, (8, 16));
BENCHMARK_P_INSTANCE(Determinant, sign_aa, (16, 32));
BENCHMARK_P_INSTANCE(Determinant, sign_aa, (32, 63));

BENCHMARK_P(Determinant, sign_aaaa, 10, 100000, (int i, int j, int a, int b)) {
    det_test.slater_sign_aaaa(i, j, a, b);
}

BENCHMARK_P_INSTANCE(Determinant, sign_aaaa, (1, 4, 32, 63));
BENCHMARK_P_INSTANCE(Determinant, sign_aaaa, (1, 4, 63, 32));
BENCHMARK_P_INSTANCE(Determinant, sign_aaaa, (63, 32, 1, 4));
