// Option 1.

class myMethod {
  public:
    myMethod() {}

    static void add_options();
};

void myMethod::add_options(Options& options) {
    /*- Compute natural orbitals using MP2 -*/
    options.add_bool("MP2_NOS", false);
}

extern "C" int read_options(std::string name, Options& options) {

    if (name == "FORTE" || options.read_globals()) {
        myMethod::add_options(options);
    }

    return true;
}
