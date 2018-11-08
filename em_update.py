from collections import Counter
from tqdm import tqdm
import numpy as np

TOPIC_SIZE = 10
LEXICON_SIZE = 51253
COLLECTION_SIZE = 18461

def em(collection: list, p_wt: np.ndarray, p_td: np.ndarray) -> (np.ndarray, np.ndarray):
  # P(Wi|Tk)
  p_wt_devisor = np.zeros(p_wt.shape[1], dtype=float) # (TOPIC_SIZE,)
  p_wt_devidend = np.zeros(p_wt.shape, dtype=float) # (51253, TOPIC_SIZE)
  for i_prime in tqdm(range(p_wt.shape[0]), desc='EM_P(Wi|Tk)'):

    # C(Wi', dj) -> dj(D
    word = str(i_prime)
    c_wd = np.zeros((1, len(collection)), dtype=int)
    for j in range(len(collection)):
      c_wd[0][j] = collection[j][word]

    # P(Tk|Wi', dj) -> dj(D
    p_twd_devisor = np.dot(p_wt[i_prime], p_td.T)
    p_twd_devidend = p_wt[i_prime] * p_td
    p_twd = (p_twd_devidend.T / p_twd_devisor).T

    # C(Wi', dj) * P(Tk|Wi', dj) -> dj(D
    sum_cp_d = np.dot(c_wd, p_twd)[0]

    p_wt_devidend[i_prime] = sum_cp_d
    p_wt_devisor += sum_cp_d
  new_wt = p_wt_devidend / p_wt_devisor

  # P(Tk|dj)
  p_td_devisor = np.zeros(len(collection), dtype=float) # (18461,)
  p_td_devidend = np.zeros(p_td.shape, dtype=float) # (18461, TOPIC_SIZE)
  for j in tqdm(range(p_td.shape[0]), desc='EM_P(Tk|dj)'):

    # C(Wi', dj) -> i=1~|V|
    c_wd = np.zeros((1, p_wt.shape[0]), dtype=int)
    for key, value in collection[j].items():
      c_wd[0][int(key)] = value
      p_td_devisor[j] += value

    # P(Tk|Wi', dj) -> i=1~|V|
    p_twd_devisor = np.dot(p_wt, p_td[j])
    p_twd_devidend = p_wt * p_td[j]
    p_twd = (p_twd_devidend.T / p_twd_devisor).T

    # C(Wi', dj) * P(Tk|Wi', dj) -> i=1~|V|
    sum_cp_v = np.dot(c_wd, p_twd)[0]

    p_td_devidend[j] = sum_cp_v
  new_td = (p_td_devidend.T / p_td_devisor).T

  return new_wt, new_td

def log_likelihood(collection: list, p_wt: np.ndarray, p_td: np.ndarray) -> float:

  likelihood = 0.
  for i in tqdm(range(p_wt.shape[0]), desc='log_likelihood'):
    # C(Wi', dj) -> dj(D
    word = str(i)
    c_wd = np.zeros(len(collection), dtype=int)
    for j in range(len(collection)):
      c_wd[j] = collection[j][word]

    # P(Wi|Tk)P(Tk|dj) -> k=1~K
    sum_pp_k = np.dot(p_wt[i], p_td.T)
    likelihood += np.dot(c_wd, np.log(sum_pp_k))

  return likelihood

def filecheck(f: np.lib.npyio.NpzFile) -> bool:
  if f["p_wt"].shape == (LEXICON_SIZE, TOPIC_SIZE) and \
     f["p_td"].shape == (COLLECTION_SIZE, TOPIC_SIZE):
    print(f["p_wt"])
    print(f["p_td"])
    return True
  return False

def main(npzout: str, npzin: str=None):
  collection = []
  with open("./resources/Collection.txt") as fileinput:
    for row in fileinput:
      collection.append(Counter(row.split()))

  if npzin is None:
    p_wt = np.random.dirichlet(np.ones(LEXICON_SIZE, dtype=float), size=TOPIC_SIZE).T
    p_td = np.random.dirichlet(np.ones(COLLECTION_SIZE, dtype=float), size=TOPIC_SIZE).T
  else:
    npzfile = np.load(npzin)
    assert filecheck(npzfile)
    p_wt = npzfile["p_wt"]
    p_td = npzfile["p_td"]

  for _ in range(5):
    p_wt, p_td = em(collection, p_wt, p_td)
    l = log_likelihood(collection, p_wt, p_td)
    print(l)
    np.savez(npzout, p_wt=p_wt, p_td=p_td)

if __name__ == "__main__":
  import sys
  import os.path
  from datetime import datetime
  assert len(sys.argv) <= 2
  npzout = "temp/" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".npz"
  if len(sys.argv) == 1:
    main(npzout)
  else:
    assert os.path.isfile(sys.argv[1])
    main(npzout, sys.argv[1])
