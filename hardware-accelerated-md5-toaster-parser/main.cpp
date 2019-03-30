#include <algorithm>
#include <string>
#include <map>
#include <fstream>
#include <vector>

using namespace std;

struct wordlist_file {
  ifstream ifs;
  int len;
  char *words;
};


// Function to generate permutations 
vector<string> permute(string input) {
    map<char, string> leet = {
        {'a', "@4"},
        {'b', "8"},
        {'e', "3"},
        {'g', "96"},
        {'i', "1"},
        {'l', "1|!"},
        {'o', "0"},
        {'s', "5$"},
        {'t', "7"},
        {'z', "2"}
    };
    vector<string> list;
    int n = input.length(); 
    // Number of permutations is 2^n 
    uint32_t max = 1 << n;
    // Using all subsequences and permuting them 
    for (int i = 0; i < max; i++) { 
        // If j-th bit is set, we convert it to upper case 
        string combination = input; 
        for (int j = 0; j < n; j++) {
            if (((i >> j) & 1) == 1 && (leet.find(input[j]) != leet.end())) {
                for (int k = 0; (leet[input[j]])[k] != '\0'; k++) {
                    combination[j] = (leet[input[j]])[k];
                }
            }     
        }
        list.push_back(combination);
    } 
    return list;
} 


map<string, int> read_wordlist(struct wordlist_file *file) {
    map<string, int> mp;
    string word;
    int len = 0;
    string lWord;
    while (file->ifs >> word) {
        char tmp [64];
        sscanf(word.c_str(), "%[a-zA-Z]", tmp);
        string newWord(tmp);
        if(len < newWord.size()){
            len = newWord.size();
            lWord = newWord;
        }
        transform(newWord.begin(), newWord.end(), newWord.begin(), ::tolower);
        if (!mp.count(newWord) && newWord.size() > 3) {
            mp.insert(make_pair(newWord, 1));
            vector<string> list = permute(newWord);
            for(int i = 0; i < list.size(); i++) {
                mp.insert(make_pair(list[i], 1));
            }
        }
    }
    printf("The longest word \"%s\" is %i long", lWord.c_str(), len);
    return mp;
}



void writeMapToFile(map<string, int> mp, const char* fn) {
    ofstream ofs;
    ofs.open(fn);
    for (auto i = mp.begin(); i != mp.end(); i++) {
        ofs << i->first << "\n";
    }
}

int main(int argc, char** argv) {

    wordlist_file words;
    words.ifs.open(argv[1]);
    map<string, int> mp;
    mp = read_wordlist(&words);
    writeMapToFile(mp, argv[2]);
}