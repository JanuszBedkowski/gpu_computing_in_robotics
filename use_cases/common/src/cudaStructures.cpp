#include "cudaStructures.h"

bool compareHashElements(const hashElement& a, const hashElement& b)
{
   return a.index_of_bucket < b.index_of_bucket;
}
