#include <math.h>

float find_distance(int x1, int y1, int x2, int y2) {
    return sqrtf(powf(x1 - x2, 2) + powf(y1 - y2, 2));
}

float hausdorff_single(int* whites, int length, int* whites2, int length2) {
  length  /= 2;
  length2 /= 2;
  float distances[length];
  float sum = 0;
  for (int i = 0; i < length; i++) {
    distances[i] = 1000.f;
    for (int j = 0; j < length2; j++) {
      float distance = find_distance(whites[i * 2], whites[i * 2 + 1], 
                                     whites2[j * 2], whites2[j * 2 + 1]);
      if (distance < distances[i]) distances[i] = distance;
    }
    sum += distances[i];
  }
  float ave = sum / (float) length;
  return ave;
}

float hausdorff(int* whites, int length, int* whites2, int length2) {
  float one = hausdorff_single(whites, length, whites2, length2);
  float two = hausdorff_single(whites2, length2, whites, length);
  return (one + two) / 2;
}

int main() {
  return 0;
}