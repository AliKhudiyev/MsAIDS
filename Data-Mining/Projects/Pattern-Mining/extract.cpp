#include <iostream>
#include <map>
#include <vector>
#include <set>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <tuple>
#include <omp.h>

using namespace std;

using eid_t = int; // element ID

struct Pattern;

using event_t = pair<int, Pattern>; // event: <timestamp, id>
using pattern_t = pair<int, Pattern>; // pattern: <timestamp, Pattern>
using dict_t = pair<eid_t, size_t>; // dict: <eventID, count>
using interval_t = pair<int, int>; // interval: <min, max>

#define CUT_FIRST   true    // command variable to drop the first event in the sequence
#define CUT_LAST    false   // command variable to drop the last event in the sequence

#define CUT_ALONE true      // status variable to know if the dropped event was alone in its corresponding pattern
#define CUT_NOT_ALONE false // status variable to know if the dropped event was not alone in its corresponding pattern

enum MiningMethod {
    GSP,
    PREFIXSPAN
};

/*
 * Pattern - a collection of one or more events
*/
struct Pattern{
    // size_t support;
    // int timestamp;
    vector<eid_t> events;
    vector<interval_t> intervals;

    /*
     * Returns pattern.events.size() if not subpattern
     * Returns index otherwise
    */
    bool is_subpattern_of(const Pattern& pattern) const{
        if(events.size() < pattern.events.size()) return pattern.events.size();

        size_t index = 0;
        for(size_t i=0; index<events.size() && i<pattern.events.size(); ++i){
            if(pattern.events[i] == events[index]) ++index;
        }
        return index == events.size()? true : false;
    }
    size_t is_superpattern_of(const Pattern& pattern) const{
        return pattern.is_subpattern_of(*this);
    }
    Pattern operator+(const eid_t& event) const{
        Pattern pattern = *this;
        pattern.events.push_back(event);
        return pattern;
    }
    Pattern& operator+=(const eid_t& event){
        (*this) = (*this) + event;
        return *this;
    }
    bool operator==(const Pattern& pattern) const{
        return events == pattern.events;
    }
    bool operator!=(const Pattern& pattern) const{
        return !((*this)==pattern);
    }
    friend ostream& operator<<(ostream& out, const Pattern& pattern){
        for(const auto& event: pattern.events)
            out << event << ' ';
        return out;
    }
};

struct Sequence{
    static eid_t global_id;

    eid_t id;
    size_t support = 0;
    vector<pattern_t> patterns;

    // Debugging-purpose variables
    static bool print_id, print_support, print_patterns, print_timestamp, print_intervals;
    static string separator;

    Sequence(bool id_0=true){
        if(!id_0)
            id = Sequence::global_id++;
        else
            id = 0;
    }
    Sequence(eid_t id_):
        id(id_)
    {
        Sequence::global_id = id+1;
    }

    bool is_subsequence_of(const Sequence& sequence) const{
        // cout << *this << " is subsequence of " << sequence << " ? \n";
        if(patterns.size() > sequence.patterns.size()) return false;
        
        size_t index = 0;
        bool result = true;
        bool found = false;
        for(size_t i=0; i<patterns.size(); ++i){
            if(index >= sequence.patterns.size()){
                result = false;
                break;
            }
            found = false;
            for(; index<sequence.patterns.size(); ++index){
                // cout << "? " << patterns[i].second << " is subpattern of " << sequence.patterns[index].second << " ? \n";
                bool t = patterns[i].second.is_subpattern_of(sequence.patterns[index].second);
                // cout << " dbg t: " << t << endl;
                if(t){
                    ++index;
                    found = true;
                    break;
                }
            }
            if(!found) return false;
            // cout << "> dbg i: " << i+1 << ", index: " << index << endl;
        }
        // cout << result << endl;
        return result;
    }

    bool is_supersequence_of(const Sequence& sequence) const{
        return sequence.is_subsequence_of(*this);
    }

    /*
     * Returns a sequence whose either the first or the last event is dropped
     * status is set true iff the dropped event was alone: <(3), (4)>
     * status is set false otherwise: <(3, 4)>
    */
    static Sequence make_sequence(const Sequence& sequence, bool cut_position, bool& status){
        Sequence result = sequence;
        size_t index = 0;
        if(cut_position == CUT_LAST){
            index = result.patterns.size()-1;
        }
        if(result.patterns[index].second.events.size() == 1){
            auto it = result.patterns.cbegin();
            result.patterns.erase(it+index);
            status = CUT_ALONE;
        } else{
            auto it = result.patterns[index].second.events.cbegin();
            result.patterns[index].second.events.erase(it+index);
            status = CUT_NOT_ALONE;
        }
        return result;
    }
    bool operator==(const Sequence& sequence) const{
        if(patterns.size() != sequence.patterns.size()) return false;
        for(size_t i=0; i<patterns.size(); ++i){
            if(patterns[i].second != sequence.patterns[i].second) return false;
        }
        return true;
    }
    friend ostream& operator<<(ostream& out, const Sequence& sequence){
        if(Sequence::print_id)
            out << sequence.id << ": ";
        if(Sequence::print_patterns){
            for(const auto& pattern: sequence.patterns){
                if(Sequence::print_timestamp)
                    out << "<" << pattern.first << "> ";
                for(size_t i=0; i<pattern.second.events.size(); ++i){
                    out << pattern.second.events[i] << " ";
                    if(Sequence::print_intervals && i<pattern.second.intervals.size())
                        out << "[" << pattern.second.intervals[i].first << "," << pattern.second.intervals[i].second << "] ";
                }
                if(!Sequence::print_timestamp)
                    out << Sequence::separator;
            }
        }
        if(sequence.print_support)
            out << "#SUP: " << sequence.support;
        return out;
    }

    static void save(const string& filepath, const vector<Sequence>& sequences, bool include_intervals){
        bool print_id = Sequence::print_id;
        bool print_timestamp = Sequence::print_timestamp;
        bool print_intervals = Sequence::print_intervals;
        Sequence::print_id = Sequence::print_timestamp = false;
        Sequence::print_intervals = include_intervals;
        if(!include_intervals)
            Sequence::separator = "| ";

        // cout << Sequence::print_id << " " << Sequence::print_timestamp << " " <<
        //     Sequence::print_intervals << " [ " << Sequence::separator << " ]\n";

        ofstream out(filepath);
        for(const auto& sequence: sequences){
            out << sequence << endl;
        }
        out.close();

        Sequence::print_id = print_id;
        Sequence::print_timestamp = print_timestamp;
        Sequence::print_intervals = print_intervals;
    }
};

eid_t Sequence::global_id = 0;
bool Sequence::print_id = true;
bool Sequence::print_patterns = true;
bool Sequence::print_timestamp = true;
bool Sequence::print_support = false;
bool Sequence::print_intervals = false;
string Sequence::separator = string();

struct DB{
    vector<Sequence> sequences;

    void init(const string& filepath);
    vector<Sequence> extract(size_t support, size_t k, MiningMethod method=MiningMethod::GSP) const;

    void save(const string& filepath) const;
    void load(const string& filepath); // not necessary

    void print(size_t n=-1, ostream& out=cout) const{
        if(n >= sequences.size()) n = sequences.size();
        for(size_t i=0; i<n; ++i){
            out << sequences[i] << endl;
        }
    }
    friend ostream& operator<<(ostream& out, const DB& db){
        db.print(db.sequences.size(), out);
        return out;
    }

    public:
    map<eid_t, size_t> all_events(size_t support=0) const{
        map<eid_t, size_t> dict;
        for(const auto& sequence: sequences){
            size_t count = 0;
            vector<eid_t> events;
            for(const auto& pattern: sequence.patterns){
                for(const auto& event: pattern.second.events)
                if(find(events.cbegin(), events.cend(), event) == events.cend()){
                    ++dict[event];
                    events.push_back(event);
                }
            }
        }
        return dict;
    }
    vector<Sequence> begin_with(const Sequence& sequence) const;
    vector<Sequence> begin_with(const Pattern& pattern) const{
        Sequence sequence;
        sequence.patterns.push_back(pair<int, Pattern>(0, pattern));
        return begin_with(sequence);
    }
    vector<Sequence> begin_with(const vector<eid_t>& events) const{
        Pattern pattern;
        pattern.events = events;
        return begin_with(pattern);
    }
    size_t count(const Sequence& sequence) const{
        return begin_with(sequence).size();
    }
    size_t count(const Pattern& pattern) const{
        Sequence sequence;
        sequence.patterns.push_back(pair<int, Pattern>(0, pattern));
        return count(sequence);
    }
    size_t support_of(const Sequence& sequence) const{
        size_t support = 0;
        // cout << " dbg seq: " << sequence << endl;
        for(const auto& s: sequences){
            if(sequence.is_subsequence_of(s)){
                ++support;
                // cout << " > subseq of " << s << endl;
            }
        }
        return support;
    }
    vector<Sequence> generate(const vector<Sequence>& sequences, size_t support=0, MiningMethod method=GSP) const{
        vector<Sequence> result;
        bool status;
        for(size_t i=0; i<sequences.size(); ++i){
            Sequence sequence = Sequence::make_sequence(sequences[i], CUT_FIRST, status);
            for(size_t j=0; j<sequences.size(); ++j){
                Sequence curr_sequence = Sequence::make_sequence(sequences[j], CUT_LAST, status);

                if(sequence == curr_sequence){
                    Sequence tmp = sequences[i];
                    eid_t e = sequences[j].patterns.back().second.events.back();
                    if(status == CUT_ALONE){
                        Pattern p;
                        p.events.push_back(e);
                        tmp.patterns.push_back(pattern_t(0, p));
                    } else{
                        tmp.patterns.back().second.events.push_back(e);
                    }
                    tmp.support = support_of(tmp);
                    // cout << " dbg: " << tmp << endl;
                    if(tmp.support >= support){
                        result.push_back(tmp);
                    }
                }
            }
        }
        return result;
    }
    pair<size_t, size_t> get_interval(const pair<Pattern, Pattern>& patterns, size_t sid) const{
        pair<size_t, size_t> interval = make_pair((size_t)-1, 0);
        const auto& db_patterns = sequences[sid].patterns;
        bool found = false;
        for(size_t i=0; i<db_patterns.size(); ++i){
            if(patterns.first != db_patterns[i].second) continue;
            for(size_t j=i+1; j<db_patterns.size(); ++j){
                if(patterns.second == db_patterns[j].second){
                    size_t dt = db_patterns[j].first - db_patterns[i].first;
                    if(!dt) continue;
                    if(interval.first > dt) interval.first = dt;
                    if(interval.second < dt) interval.second = dt;
                    found = true;

                    // if(!dt) cout << ">>> " << db_patterns[j].first << endl;
                }
            }
        }
        if(!found) interval.first = interval.second = 0;
        // if(interval.first == 0){
        //     cout << "> sid:" << sid << ", found: " << found << endl;
        // }
        return interval;
    }
    void set_intervals(vector<Sequence>& sequences) const{
        vector<tuple<Pattern, Pattern, size_t, size_t>> dict;
        for(auto& sequence: sequences){
            auto& patterns = sequence.patterns;
            size_t min = -1, max = 0;
            for(size_t i=0; i<patterns.size()-1; ++i, min=-1, max=0){
                pair<Pattern, Pattern> pattern_couple = make_pair(patterns[i].second, patterns[i+1].second);
                auto it = find_if(dict.cbegin(), dict.cend(), 
                                [pattern_couple](const tuple<Pattern, Pattern, size_t, size_t>& tuple_){
                                    return (get<0>(tuple_) == pattern_couple.first && get<1>(tuple_) == pattern_couple.second);
                                });
                if(it != dict.cend()){
                    patterns[i].second.intervals.push_back(make_pair(get<2>(*it), get<3>(*it)));
                    // cout << "already exists\n";
                    continue;
                }
                for(size_t sid=0; sid<sequences.size(); ++sid){
                    auto tmp = get_interval(pair<Pattern, Pattern>(patterns[i].second, patterns[i+1].second), sid);
                    // cout << " dbg: " << tmp.first << "," << tmp.second << endl;
                    if(!tmp.first && !tmp.second) continue;
                    if(min > tmp.first) min = tmp.first;
                    if(max < tmp.second) max = tmp.second;
                }
                dict.push_back(make_tuple(patterns[i].second, patterns[i+1].second, min, max));
                patterns[i].second.intervals.push_back(make_pair(min, max));
                // cout << "+ " << min << ", " << max << " | " << patterns[i].first << " and " << patterns[i+1].first << "\n";
            }
        }
    }
};

int main(int argc, char** argv){
    if(argc < 6){
        cerr << "Usage: ./extract [input] [support] [k] [method] [output] [include_intervals(opt)]\n";
        exit(0);
    }
    
    size_t support = stoul(argv[2]), k = stoul(argv[3]);
    string m(argv[4]), filepath_out(argv[5]), filepath_in(argv[1]);
    bool print_intervals = false;
    MiningMethod method;
    if(m == "gsp") method = MiningMethod::GSP;
    else if(m == "prefixspan") method = MiningMethod::PREFIXSPAN;
    else{
        cerr << "No such method to mine!" << endl;
        // return 1;
    }
    if(argc == 7) print_intervals = true;

    Sequence s1, s2;
    Pattern p11, p12, p21, p22, p23;
    p11.events.push_back(1);
    p12.events.push_back(1);
    p21.events.push_back(1);
    p22.events.push_back(2);
    p23.events.push_back(3);
    s1.patterns.push_back(pattern_t(0, p11));
    s1.patterns.push_back(pattern_t(0, p12));
    s2.patterns.push_back(pattern_t(0, p21));
    s2.patterns.push_back(pattern_t(0, p22));
    s2.patterns.push_back(pattern_t(0, p23));

    // Sequence::print_id = false;
    // cout << s1.is_subsequence_of(s2) << endl;

    DB db;
    db.init(filepath_in);
    // db.print(5);
    // auto dict = db.all_events();
    // for(auto it=dict.cbegin(); it!=dict.cend(); ++it){
    //     cout << it->first << ": " << it->second << endl;
    // }
    // db.save(filepath_out);
    Sequence::print_support = true;
    vector<Sequence> sequences = db.extract(support, k, method);
    db.set_intervals(sequences);
    cout << "Result:\n";
    for(const auto& seq: sequences)
        cout << seq << endl;
    Sequence::save(filepath_out, sequences, print_intervals);

    /*
    Sequence::print_support = true;
    vector<Sequence> sequences;

    Sequence s1;
    Pattern p11, p12;
    p11.events.push_back(1);
    p11.events.push_back(2);
    p12.events.push_back(3);
    s1.patterns.push_back(pattern_t(0, p11));
    s1.patterns.push_back(pattern_t(0, p12));

    Sequence s2;
    Pattern p21, p22;
    p21.events.push_back(1);
    p21.events.push_back(2);
    p22.events.push_back(4);
    s2.patterns.push_back(pattern_t(0, p21));
    s2.patterns.push_back(pattern_t(0, p22));

    Sequence s3;
    Pattern p31, p32;
    p31.events.push_back(1);
    p32.events.push_back(3);
    p32.events.push_back(4);
    s3.patterns.push_back(pattern_t(0, p31));
    s3.patterns.push_back(pattern_t(0, p32));

    Sequence s4;
    Pattern p41, p42;
    p41.events.push_back(1);
    p41.events.push_back(3);
    p42.events.push_back(5);
    s4.patterns.push_back(pattern_t(0, p41));
    s4.patterns.push_back(pattern_t(0, p42));

    Sequence s5;
    Pattern p51, p52;
    p51.events.push_back(2);
    p52.events.push_back(3);
    p52.events.push_back(4);
    s5.patterns.push_back(pattern_t(0, p51));
    s5.patterns.push_back(pattern_t(0, p52));

    Sequence s6;
    Pattern p61, p62, p63;
    p61.events.push_back(2);
    p62.events.push_back(3);
    p63.events.push_back(5);
    s6.patterns.push_back(pattern_t(0, p61));
    s6.patterns.push_back(pattern_t(0, p62));
    s6.patterns.push_back(pattern_t(0, p63));

    vector<Sequence> seqs{
        s1, s2, s3, s4, s5, s6
    };
    cout << s1 << '\n' << s2 << '\n' << s3 << '\n' << s4 << '\n' << s5 << '\n' << s6 << endl;
    cout << endl;
    vector<Sequence> result = db.generate(seqs);
    for(const auto& sequence: result){
        cout << sequence << endl;
    }
    */

    return 0;
}

void DB::init(const string& filepath){
    ifstream in(filepath);
    string line;
    eid_t id = 0;
    while(getline(in, line)){
        stringstream stream(line);
        string token;
        Sequence sequence { id++ };
        while(getline(stream, token, ',')){
            event_t event;
            size_t index = find(token.cbegin(), token.cend(), ' ') - token.cbegin();
            event.first = stoul(token.substr(1, index));
            event.second.events.push_back(stoi(token.substr(index+1)));

            Pattern pattern;
            pattern.events.push_back(stoi(token.substr(index+1)));
            sequence.patterns.push_back(pair<int, Pattern>(event.first, pattern));
            // sequence.events.push_back(event);
        }
        sequences.push_back(sequence);
    }
    in.close();
}

vector<Sequence> DB::extract(size_t support, size_t k, MiningMethod method) const{
    vector<Sequence> frequent_sequences;
    vector<Sequence> result;
    map<eid_t, size_t> dict = all_events();

    for(auto it=dict.cbegin(); it!=dict.cend(); ++it){
        if(it->second >= support){
            Sequence s;
            Pattern p;
            s.support = it->second;
            // p.support = it->second;
            p += it->first;
            s.patterns.push_back(pattern_t(0, p));
            result.push_back(s);
        }
    }

    size_t length=1;
    while(!result.empty() && length++ < k){
        // cout << "- - - - -\n";
        for(const auto& seq: result){
            frequent_sequences.push_back(seq);
            // cout << seq << endl;
        }
        // cout << "- - - - -\n";
        result = generate(result, support, method);
    }

    return frequent_sequences;
}

void DB::save(const string& filepath) const{
    ofstream out(filepath);
    for(const auto& sequence: sequences){
        for(const auto& pattern: sequence.patterns){
            out << "<" << pattern.first << ">";
            for(const auto& event: pattern.second.events){
                out << " " << event+1;
            }   out << " -1 ";
        }   out << "-2\n";
    }
    out.close();
}

vector<Sequence> DB::begin_with(const Sequence& sequence) const{
    vector<Sequence> result;

    for(size_t i=0, pi=0; i<sequences.size(); ++i, ++pi){
        const auto& sequence_ = sequences[i];
        bool found = true;
        Sequence seq;
        for(size_t j=0, ei=0; j<sequence.patterns.size(); ++j){
            const auto& events = sequence.patterns[j].second.events;
            const auto& pattern_ = sequence_.patterns[pi];
            const auto& events_ = pattern_.second.events;
            bool pattern_found = true;
            for(size_t t=0; t<events.size() && ei<events_.size(); ++t, ++ei){
                if(events[t] != events_[ei] || events_.size()-ei < events.size()-t){
                    pattern_found = false;
                    break;
                }

            }
            if(!pattern_found) --j;
            else{
                seq.patterns.push_back(pair<int, Pattern>(0, sequence.patterns[j].second));
            }
            if(ei == events_.size()){
                found = false;
                break;
            }
        }
        if(found){
            result.push_back(seq);
        }
    }
    return result;
}