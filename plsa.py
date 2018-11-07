# dev_dependencies
from tqdm import tqdm
from sys import getsizeof
# dependencies
import gc
from collections import Counter
import numpy as np

TOPIC_SIZE = 256
LEXICON_SIZE = 51253
COLLECTION_SIZE = 18461

def m_step(collection: list, p_wt: np.ndarray, p_td: np.ndarray):
  old_wt = p_wt.copy(); old_td = p_td.copy()

  p_wt_devisor = np.zeros(old_wt.shape[1], dtype=float)
  p_wt_dividend = np.zeros(old_wt.shape, dtype=float)
  for i_prime in tqdm(range(old_wt.shape[0])):
  # for i_prime in range(old_wt.shape[0]):

    # C(Wi', dj), dj(D
    word = str(i_prime)
    c_wd = np.zeros((1, len(collection)), dtype=int)
    for j in range(len(collection)):
      c_wd[0][j] = collection[j][word]

    # P(Tk|Wi', dj), dj(D
    p_twd_devisor = np.dot(old_wt[i_prime], old_td.T)
    p_twd_devidend = old_wt[i_prime] * old_td
    p_twd = (p_twd_devidend.T / p_twd_devisor).T
    sum_cp_d = np.dot(c_wd, p_twd)[0]
    # sum_cp_v =
    p_wt_devisor += sum_cp_d

  p_wt = p_wt_dividend / p_wt_devisor
  # for i in range(p_wt.shape[0]):




def main():
  collection = []
  with open("./Collection.txt") as fileinput:
    for row in fileinput:
      collection.append(Counter(row.split()))

  p_wt = np.random.dirichlet(np.ones(LEXICON_SIZE, dtype=float), size=TOPIC_SIZE).T
  p_td = np.random.dirichlet(np.ones(COLLECTION_SIZE, dtype=float), size=TOPIC_SIZE).T

  for _ in range(1):
    m_step(collection, p_wt, p_td)
  # with open("./BGLM.txt") as fileinput:
  #   content = fileinput.read().split()
  # BGLM = np.array(content[1::2])

if __name__ == "__main__":
  main()
