from collections import Counter
from tqdm import tqdm
import cupy as cp
import numpy as np

TOPIC_SIZE = 256
LEXICON_SIZE = 51253
COLLECTION_SIZE = 18461
UPDATE_THRESHOLD = 1e-4

def em(collection: list, p_wt: cp.ndarray, p_td: cp.ndarray) -> (cp.ndarray, cp.ndarray):
  # P(Wi|Tk)
  p_wt_devisor = cp.zeros(p_wt.shape[1], dtype=float) # (TOPIC_SIZE,)
  p_wt_devidend = cp.zeros(p_wt.shape, dtype=float) # (51253, TOPIC_SIZE)
  for i_prime in tqdm(range(p_wt.shape[0]), desc='EM_P(Wi|Tk)'):

    # C(Wi', dj) -> dj(D
    word = str(i_prime)
    c_wd = [[collection[j][word] for j in range(len(collection))]]

    # P(Tk|Wi', dj) -> dj(D
    p_twd_devisor = cp.dot(p_wt[i_prime], p_td.T)
    p_twd_devidend = p_wt[i_prime] * p_td
    p_twd = cp.nan_to_num(p_twd_devidend.T / p_twd_devisor).T

    # C(Wi', dj) * P(Tk|Wi', dj) -> dj(D
    sum_cp_d = cp.dot(c_wd, p_twd)[0]

    p_wt_devidend[i_prime] = sum_cp_d
    p_wt_devisor += sum_cp_d
  new_wt = p_wt_devidend / p_wt_devisor

  # P(Tk|dj)
  p_td_devisor = cp.zeros(len(collection), dtype=float) # (18461,)
  p_td_devidend = cp.zeros(p_td.shape, dtype=float) # (18461, TOPIC_SIZE)
  for j in tqdm(range(p_td.shape[0]), desc='EM_P(Tk|dj)'):

    # C(Wi', dj) -> i=1~|V|
    c_wd = cp.zeros((1, p_wt.shape[0]), dtype=int)
    for key, value in collection[j].items():
      c_wd[0][int(key)] = value
      p_td_devisor[j] += value

    # P(Tk|Wi', dj) -> i=1~|V|
    p_twd_devisor = cp.dot(p_wt, p_td[j])
    p_twd_devidend = p_wt * p_td[j]
    p_twd = cp.nan_to_num(p_twd_devidend.T / p_twd_devisor).T

    # C(Wi', dj) * P(Tk|Wi', dj) -> i=1~|V|
    sum_cp_v = cp.dot(c_wd, p_twd)[0]

    p_td_devidend[j] = sum_cp_v
  new_td = (p_td_devidend.T / p_td_devisor).T

  return new_wt, new_td

def log_likelihood(collection: list, p_wt: cp.ndarray, p_td: cp.ndarray) -> float:

  likelihood = 0.
  for i in tqdm(range(p_wt.shape[0]), desc='Log_like'):
    # C(Wi', dj) -> dj(D
    word = str(i)
    c_wd = [collection[j][word] for j in range(len(collection))]

    # P(Wi|Tk)P(Tk|dj) -> k=1~K
    sum_pp_k = cp.dot(p_wt[i], p_td.T)
    likelihood += cp.dot(c_wd, cp.nan_to_num(cp.log(sum_pp_k)))

  return likelihood

def filecheck(f: np.lib.npyio.NpzFile) -> bool:
  if f["p_wt"].shape == (LEXICON_SIZE, TOPIC_SIZE) and \
     f["p_td"].shape == (COLLECTION_SIZE, TOPIC_SIZE):
    print(f["p_wt"])
    print(f["p_td"])
    return True
  return False

def normalize(cparray: cp.ndarray, axis=-1) -> cp.ndarray:
  return cparray / cparray.sum(axis=axis, keepdims=True)

def main(npzout: str, npzin: str=None):
  collection = []
  with open("./resources/Collection.txt") as fileinput:
    for row in fileinput:
      collection.append(Counter(row.split()))

  if npzin is None:
    p_wt = normalize(cp.random.rand(LEXICON_SIZE, TOPIC_SIZE), axis=0)
    p_td = normalize(cp.random.rand(COLLECTION_SIZE, TOPIC_SIZE), axis=0)
  else:
    npzfile = cp.load(npzin)
    assert filecheck(npzfile)
    p_wt = npzfile["p_wt"]
    p_td = npzfile["p_td"]

  baseline_likelihood = log_likelihood(collection, p_wt, p_td)
  print("likelihood(baseline): %f" %(baseline_likelihood))
  prev_likelihood = baseline_likelihood
  for index in range(1000):
    p_wt, p_td = em(collection, p_wt, p_td)
    cp.savez(npzout, p_wt=p_wt, p_td=p_td)
    likelihood = log_likelihood(collection, p_wt, p_td)
    print("likelihood(%d): %f" %(index+1, likelihood))
    if (likelihood - prev_likelihood) / abs(prev_likelihood) < UPDATE_THRESHOLD:
      break
    prev_likelihood = likelihood

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
