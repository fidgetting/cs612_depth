
#ifndef DISJOIN_SET_H_
#define DISJOIN_SET_H_

typedef struct {
  int rank;
  int p;
  int size;
} uni_elt;

class universe {
public:
  universe(int elements);
  ~universe();
  int find(int x);
  void join(int x, int y);
  inline int size(int x) const { return elts[x].size; }
  inline int num_sets() const { return num; }

private:
  uni_elt *elts;
  int num;
};


#endif /* DISJOIN_SET_H_ */
