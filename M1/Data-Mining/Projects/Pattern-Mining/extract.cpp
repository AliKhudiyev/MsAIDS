#include "extract.hpp"

int main(int argc, char** argv){
    if(argc < 6){
        cerr << "Usage: ./extract [input] [support] [max_length] [method] [output] [include_intervals(opt)]\n";
        exit(0);
    }
    
    float support = stof(argv[2]);
    size_t k = stoul(argv[3]);
    string m(argv[4]), filepath_out(argv[5]), filepath_in(argv[1]);
    bool print_intervals = false;
    MiningMethod method;

    if(m == "gsp") method = MiningMethod::GSP;
    else if(m == "prefixspan") method = MiningMethod::PREFIXSPAN;
    else{
        cerr << "No such method to mine!" << endl;
        return 1;
    }
    if(support <= 0 || support > 1){
        cerr << "Invalid support!" << endl;
        return 1;
    }
    if(argc == 7) print_intervals = true;

    DB db;
    db.init(filepath_in);
    // db.print(5);
    db.save("out.txt");
    Sequence::print_support = true;
    vector<Sequence> sequences = db.extract(support*db.size(), k, method, print_intervals);
    // cout << "Result:\n";
    // for(const auto& seq: sequences)
    //     cout << seq << endl;
    Sequence::save(filepath_out, sequences, print_intervals);

    return 0;
}