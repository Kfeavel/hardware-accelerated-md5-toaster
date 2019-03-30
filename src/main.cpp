#include "main.h"
#include <algorithm>
#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <netinet/in.h>
#include <string.h>

#define MD5_INPUT_LENGTH 512

using namespace std;

extern void md5_calculate(struct cuda_device *);

char *md5_unpad(char *input)
{
  static char md5_unpadded[MD5_INPUT_LENGTH];
  unsigned int orig_length;
  int x;

  if (input == NULL)
  {
    return NULL;
  }

  memset(md5_unpadded, 0, sizeof(md5_unpadded));

  orig_length = (*((unsigned int *)input + 14) / 8);

  strncpy(md5_unpadded, input, orig_length);

  return md5_unpadded;
}



int get_cuda_device(struct cuda_device *device)
{
  int device_count;

  if (cudaGetDeviceCount(&device_count) != CUDA_SUCCESS)
  {
    // cuda not supported
    return -1;
  }

  while (device_count >= 0)
  {
    if (cudaGetDeviceProperties(&device->prop, device_count) == CUDA_SUCCESS)
    {
      // we have found our device
      device->device_id = device_count;
      return device_count;
    }

    device_count--;
  }

  return -1;
}

#define REQUIRED_SHARED_MEMORY 64
#define FUNCTION_PARAM_ALLOC 256

int calculate_cuda_params(struct cuda_device *device)
{
  int max_threads;
  int max_blocks;
  int shared_memory;

  max_threads = device->prop.maxThreadsPerBlock;
  shared_memory = device->prop.sharedMemPerBlock - FUNCTION_PARAM_ALLOC;

  // calculate the most threads that we can support optimally

  while ((shared_memory / max_threads) < REQUIRED_SHARED_MEMORY)
  {
    max_threads--;
  }

  // now we spread our threads across blocks

  max_blocks = 40; // ?? we need to calculate this !

  device->max_threads = max_threads;     // most threads we support
  device->max_blocks = max_blocks;       // most blocks we support
  device->shared_memory = shared_memory; // shared memory required

  // now we need to have (device.max_threads * device.max_blocks) number of
  // words in memory for the graphics card

  device->device_global_memory_len =
      (device->max_threads * device->max_blocks) * 64;

  return 1;
}

struct wordlist_file
{
  std::ifstream ifs;
  int len;
  char *words;
};

#define WORDS_TO_CACHE 10000
#define FILE_BUFFER 512

#define CRLF 2
#define LF 1

int read_wordlist(struct wordlist_file *file)
{

  std::string word;
  vector<string> words;
  while (file->ifs >> word)
  {
    words.push_back(word);
  }
  file->len = 0;
  cudaMallocManaged(&(file->words), words.size() * 64 * sizeof(char));
  for (int i = 0; i < words.size(); i++) {
	char *tmp = word.c_str();
    copy(&tmp, &tmp + 64, file->words + file.len * 64);
    file->len++;
  }


  return 1;
}

int read_hashlist(ifstream ifs, struct cuda_device *device)
{
  vector<int[4]> hashes;
  std::string hash;
  int len = 0;
  while (ifs >> hash)
  {
    hashes.push_back({0, 0, 0, 0});
    sscanf(hash.c_str(), "%x%x%x%x", hashes[len][0], hashes[len * 4][1],
           hashes[len * 4][2], hashes[len * 4][3]);
    len++;
  }
  cudaMallocManaged(&(device.target_hash), (len + 1) * 4 * sizeof(int));
  for (int i = 0; i < len; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      (reinterpret_cast<int(*)[4]>(device.target_hash))[i][j] = hashes[i][j];
    }
  }
}

/*********************************************************************
 *  TAKEN FROM: http://www.codeproject.com/KB/string/hexstrtoint.aspx
 *
 *  has been slightly modified
 *
 *  Many Thanks Anders Molin
 *********************************************************************/

struct CHexMap
{
  char chr;
  int value;
};

#define true 1
#define false 0

#define HexMapL 22

int _httoi(const char *value)
{
  struct CHexMap HexMap[HexMapL] = {
      {'0', 0},
      {'1', 1},
      {'2', 2},
      {'3', 3},
      {'4', 4},
      {'5', 5},
      {'6', 6},
      {'7', 7},
      {'8', 8},
      {'9', 9},
      {'A', 10},
      {'B', 11},
      {'C', 12},
      {'D', 13},
      {'E', 14},
      {'F', 15},
      {'a', 10},
      {'b', 11},
      {'c', 12},
      {'d', 13},
      {'e', 14},
      {'f', 15},
  };
  int i;

  char *mstr = strdup(value);
  char *s = mstr;
  int result = 0;
  int found = false;

  if (*s == '0' && *(s + 1) == 'X')
  {
    s += 2;
  }

  int firsttime = true;

  while (*s != '\0')
  {
    for (i = 0; i < HexMapL; i++)
    {

      if (*s == HexMap[i].chr)
      {

        if (!firsttime)
        {
          result <<= 4;
        }

        result |= HexMap[i].value;
        found = true;
        break;
      }
    }

    if (!found)
    {
      break;
    }

    s++;
    firsttime = false;
  }

  free(mstr);
  return result;
}

/*************************************************************************/

void print_info(void)
{
  printf("cuda_md5_crack programmed by XPN "
         "(http://xpnsbraindump.blogspot.com)\n\n");
  return;
}

#define ARG_MD5 2
#define ARG_WORDLIST 1
#define ARG_COUNT 1 + 2

int main(int argc, char **argv)
{
  char *output;
  int x;
  int y;
  struct cuda_device device;
  int available_words = 1;
  int current_words = 0;
  struct wordlist_file file;
  char input_hash[4][9];

  print_info();

  if (argc != ARG_COUNT)
  {
    printf("Usage: %s WORDLIST_FILE MD5_HASH\n", argv[0]);
    return -1;
  }
  file.ifs.open(argv[ARG_WORDLIST]);
  if (file.ifs.is_open())
  {
    printf("Error Opening Wordlist File: %s\n", argv[ARG_WORDLIST]);
    return -1;
  }

  if (read_wordlist(&file) == 0)
  {
    printf("Words in the wordlist file: %s\n", argv[ARG_WORDLIST]);
    return -1;
  }
  else
  {
    printf("%s contains %i words", argv[ARG_WORDLIST], file.len);
  }
  file.ifs.close();

  // first things first, we need to select our CUDA device

  if (get_cuda_device(&device) == -1)
  {
    printf("No Cuda Device Installed\n");
    return -1;
  }

  // we now need to calculate the optimal amount of threads to use for this card

  calculate_cuda_params(&device);

  // now we input our target hash
  ifstream ifs;
  ifs.open(argv[ARG_MD5]);
  if (!ifs.is_open())
  {
    printf("Error Opening Hashlist File: %s\n", argv[ARG_MD5]);
    return -1;
  }

  read_hashlist(ifs, &device);
  ifs.close();

  // we split the input hash into 4 blocks
  cudaMallocManaged(&(device.device_stats_memory), sizeof(device_stats));
  copy(&(device.device_stats), &(device.device_stats) + sizeof(device_stats), device.device_stats_memory);

  md5_calculate(&device); // launch the kernel of the CUDA device



  if (device.stats.hash_found != 1)
  {
    printf("No word could be found for the provided MD5 hash\n");
  }
}
