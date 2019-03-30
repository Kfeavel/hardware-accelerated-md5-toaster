#include <algorithm>
#include <string>
#include <map>
#include <fstream>

using namespace std;

struct wordlist_file {
  ifstream ifs;
  int len;
  char *words;
};

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
        } else if(newWord.size() > 3) {
            mp[newWord]++;
        }
    }
    printf("The longest word\"%s\" is %i long", lWord.c_str(), len);
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