#include <math.h>
#include <stdlib.h>

double find_distance(int x1, int y1, int x2, int y2) {
  return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

double hausdorff_single(int* whites, int length, int* whites2, int length2) {
  length  /= 2;
  length2 /= 2;
  double distances[length];
  double sum = 0;
  for (int i = 0; i < length; i++) {
    distances[i] = 1000.f;
    for (int j = 0; j < length2; j++) {
      double distance = find_distance(whites[i * 2], whites[i * 2 + 1], 
                                     whites2[j * 2], whites2[j * 2 + 1]);
      if (distance < distances[i]) distances[i] = distance;
    }
    sum += distances[i];
  }
  double ave = sum / (double) length;
  return ave;
}

double hausdorff(int* whites, int length, int* whites2, int length2) {
  double one = hausdorff_single(whites, length, whites2, length2);
  double two = hausdorff_single(whites2, length2, whites, length);
  double ans = (one + two) / 2;
  return (one + two) / 2;
}

int main() {
  return 0;
}