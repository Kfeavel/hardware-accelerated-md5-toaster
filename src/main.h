#ifndef main_h
#define main_h
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fstream>
using namespace std;

struct device_stats {
  unsigned char word[32][64]; // found word passed from GPU
  int hash[32];
  int hash_found;         // boolean if word is found
};


struct wordlist_file
{
  ifstream ifs;
  int len;
  char *words;
};


struct cuda_device {
  int device_id;
  struct cudaDeviceProp prop;

  int max_threads;
  int max_blocks;
  int shared_memory;

  void *wordlist;
  int wordlist_len;

  void *host_memory;

  void *device_stats_memory;
  struct device_stats stats;

  unsigned int *target_hash;
  int targets;

  // to be used for debugging
  void *device_debug_memory;
};

#endif