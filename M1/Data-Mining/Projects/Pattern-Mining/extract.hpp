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

using pattern_t = pair<int, Pattern>;   // pattern: <timestamp, Pattern>
using dict_t = pair<eid_t, size_t>;     // dict: <eventID, count>
using interval_t = pair<int, int>;      // interval: <min, max>

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
 * A pattern only contains only the events that have the same timestamp
*/
struct Pattern{
    vector<eid_t> events;
    vector<interval_t> intervals;

    /*
     * Returns true, if the caller object is subpatern of the given [pattern]
     * Returns false, otherwise
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

/*
 * Sequence - a collection of one or more patterns
 * A sequence has also a unique SID and support
*/
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

    /*
     * Returns true, if the caller object is subsequence of the give [sequence]
     * Returns false, otherwise
    */
    bool is_subsequence_of(const Sequence& sequence) const{
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
                if(patterns[i].second.is_subpattern_of(sequence.patterns[index].second)){
                    ++index;
                    found = true;
                    break;
                }
            }
            if(!found) return false;
        }
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

class DB{
    private:
    vector<Sequence> sequences;

    public:
    void init(const string& filepath);
    inline size_t size() const{ return sequences.size(); }
    vector<Sequence> extract(size_t support, size_t k, MiningMethod method=MiningMethod::GSP, bool include_intervals=false) const;

    void save(const string& filepath) const;
    // void load(const string& filepath);

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

    private:
    // Return all 1-itemsets of the form map<eventID, support>
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
        for(const auto& s: sequences){
            if(sequence.is_subsequence_of(s)){
                ++support;
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
                    // if(!dt) continue;
                    if(interval.first > dt) interval.first = dt;
                    if(interval.second < dt) interval.second = dt;
                    found = true;
                }
            }
        }
        if(!found) interval.first = interval.second = 0;
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
                    continue;
                }
                for(size_t sid=0; sid<sequences.size(); ++sid){
                    auto tmp = get_interval(pair<Pattern, Pattern>(patterns[i].second, patterns[i+1].second), sid);
                    if(!tmp.first && !tmp.second) continue;
                    if(min > tmp.first) min = tmp.first;
                    if(max < tmp.second) max = tmp.second;
                }
                dict.push_back(make_tuple(patterns[i].second, patterns[i+1].second, min, max));
                patterns[i].second.intervals.push_back(make_pair(min, max));
            }
        }
    }
};

/*
 * Loades the database from the given file
 * File format should comly with the one in 'sequences.csv'
*/
void DB::init(const string& filepath){
    ifstream in(filepath);
    string line;
    eid_t id = 0;
    while(getline(in, line)){
        stringstream stream(line);
        string token;
        Sequence sequence { id++ };
        while(getline(stream, token, ',')){
            size_t index = find(token.cbegin(), token.cend(), ' ') - token.cbegin();
            size_t timestamp = stoul(token.substr(1, index));
            int event = stoi(token.substr(index+1));
            size_t i = find_if(sequence.patterns.cbegin(), sequence.patterns.cend(), 
                                [timestamp](const pattern_t& pattern){
                                    return pattern.first == timestamp;
                                }) - sequence.patterns.cbegin();
            if(i == sequence.patterns.size()){
                Pattern pattern;
                pattern.events.push_back(event);
                sequence.patterns.push_back(pattern_t(timestamp, pattern));
            } else{
                sequence.patterns[i].second.events.push_back(event);
            }
        }
        sequences.push_back(sequence);
    }
    in.close();
}

/*
 * Extracts/mines sequential frequent itemsets
 * support - minimum support threshold (%)
 * k - maximum itemset length
 * method - algorithm to mine with
 * include_intervals - to extract time intervals between patterns
*/
vector<Sequence> DB::extract(size_t support, size_t k, MiningMethod method, bool include_intervals) const{
    vector<Sequence> frequent_sequences;
    vector<Sequence> result;                    // holds generated frequent-itemsets in a level-wise manner
    map<eid_t, size_t> dict = all_events();     // to get all 1-itemsets

    // pruning the 1-itemsets to get frequent 1-itemsets
    for(auto it=dict.cbegin(); it!=dict.cend(); ++it){
        if(it->second >= support){
            Sequence s;
            Pattern p;
            s.support = it->second;
            p += it->first;
            s.patterns.push_back(pattern_t(0, p));
            result.push_back(s);
        }
    }

    size_t length=1;
    while(!result.empty() && length++ < k){
        for(const auto& seq: result){               // save all frequent [length]-itemsets
            frequent_sequences.push_back(seq);
        }
        result = generate(result, support, method); // generate frequent [length+1]-itemsets
    }

    if(include_intervals) set_intervals(frequent_sequences);
    return frequent_sequences;
}

/*
 * Saves database in a standartized format (i.e. to be used in SPMF)
*/
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

/*
 * Returns all sequences in database that has the given prefix [sequence]
*/
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