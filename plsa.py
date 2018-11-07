# dev_dependencies
from tqdm import tqdm
from sys import getsizeof
# dependencies
import gc
from collections import Counter
import numpy as np

TOPIC_SIZE = 10
LEXICON_SIZE = 51253
COLLECTION_SIZE = 18461

def m_step(collection: list, p_wt: np.ndarray, p_td: np.ndarray):
  old_wt = p_wt.copy(); old_td = p_td.copy()

  p_wt_devisor = np.zeros(old_wt.shape[1], dtype=float) # (TOPIC_SIZE,)
  p_wt_devidend = np.zeros(old_wt.shape, dtype=float) # (51253, TOPIC_SIZE)
  # for i_prime in tqdm(range(old_wt.shape[0])):
  for i_prime in range(old_wt.shape[0]):

    # C(Wi', dj) -> dj(D
    word = str(i_prime)
    c_wd = np.zeros((1, len(collection)), dtype=int)
    for j in range(len(collection)):
      c_wd[0][j] = collection[j][word]

    # P(Tk|Wi', dj) -> dj(D
    p_twd_devisor = np.dot(old_wt[i_prime], old_td.T)
    p_twd_devidend = old_wt[i_prime] * old_td
    p_twd = (p_twd_devidend.T / p_twd_devisor).T

    # C(Wi', dj) * P(Tk|Wi', dj) -> dj(D
    sum_cp_d = np.dot(c_wd, p_twd)[0]

    p_wt_devidend[i_prime] = sum_cp_d
    p_wt_devisor += sum_cp_d
  p_wt = p_wt_devidend / p_wt_devisor

  p_td_devisor = np.zeros(len(collection), dtype=float) # (18461,)
  p_td_devidend = np.zeros(old_td.shape, dtype=float) # (18461, TOPIC_SIZE)
  # for j in tqdm(range(old_td.shape[0])):
  for j in range(old_td.shape[0]):

    # C(Wi', dj) -> i=1~|V|
    c_wd = np.zeros((1, old_wt.shape[0]), dtype=int)
    for key, value in collection[j].items():
      c_wd[0][int(key)] = value
      p_td_devisor[j] += value

    # P(Tk|Wi', dj) -> i=1~|V|
    p_twd_devisor = np.dot(old_wt, old_td[j])
    p_twd_devidend = old_wt * old_td[j]
    p_twd = (p_twd_devidend.T / p_twd_devisor).T

    # C(Wi', dj) * P(Tk|Wi', dj) -> i=1~|V|
    sum_cp_v = np.dot(c_wd, p_twd)[0]

    p_td_devidend[j] = sum_cp_v
  p_td = (p_td_devidend.T / p_td_devisor).T

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
