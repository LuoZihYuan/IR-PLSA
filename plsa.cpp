#include <fstream>
#include <iostream>
#include <string>
using namespace std;
int main(void) {
  ifstream fileinput;
  fileinput.open("./BGLM.txt");
  if (fileinput.is_open()) {
    string row;
    while(getline(fileinput, row)) {
      cout << row << endl;
    }
    fileinput.close();
  }
}
