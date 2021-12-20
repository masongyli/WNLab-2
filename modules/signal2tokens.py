import csv as csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import argrelextrema

DELIMITER_DA = [-1, -1, -1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1, -1,  1]
DELIMITER_DB = [ 1,  1,  1, -1, -1, -1,  1, -1, -1, -1,  1, -1, -1,  1, -1]
DELIMITER_FA = [-1, -1, -1, -1,  1, -1,  1, -1, -1,  1,  1, -1, -1,  1]
DELIMITER_FB = [ 1,  1,  1,  1, -1,  1, -1,  1,  1, -1, -1,  1,  1, -1]

AUTOCORRELATION_LENGTH = 90
HL_LENGTH = 6

BIT_0_EXTEND = []
for _ in range(AUTOCORRELATION_LENGTH // 2):
  BIT_0_EXTEND.append(1)
for _ in range(AUTOCORRELATION_LENGTH // 2):
  BIT_0_EXTEND.append(-1)

BIT_1_EXTEND = []
for _ in range(AUTOCORRELATION_LENGTH // 2):
  BIT_1_EXTEND.append(-1)
for _ in range(AUTOCORRELATION_LENGTH // 2):
  BIT_1_EXTEND.append(1)

DELIMITER_DA_EXTEND = []
for i in DELIMITER_DA:
  for _ in range(HL_LENGTH):
    DELIMITER_DA_EXTEND.append(DELIMITER_DA[i])

DELIMITER_DB_EXTEND = []
for i in DELIMITER_DB:
  for _ in range(HL_LENGTH):
    DELIMITER_DB_EXTEND.append(DELIMITER_DB[i])

DELIMITER_FA_EXTEND = []
DELIMITER_FA_EXTEND.extend([-1, -1, -1])
for i in DELIMITER_FA:
  for _ in range(HL_LENGTH):
    DELIMITER_FA_EXTEND.append(DELIMITER_FA[i])
DELIMITER_FA_EXTEND.extend([1, 1, 1])

DELIMITER_FB_EXTEND = []
DELIMITER_FB_EXTEND.extend([1, 1, 1])
for i in DELIMITER_FB:
  for _ in range(HL_LENGTH):
    DELIMITER_FB_EXTEND.append(DELIMITER_FB[i])
DELIMITER_FB_EXTEND.extend([-1, -1, -1])

TOKEN_EXTEND = [BIT_0_EXTEND, BIT_1_EXTEND, DELIMITER_DA_EXTEND, DELIMITER_DB_EXTEND, DELIMITER_FA_EXTEND, DELIMITER_FB_EXTEND] 

def signal2tokens(signals:list) -> list:
  token_sequences = []
  for idx, signal in enumerate(tqdm(signals)):
    autocorrelations = []
    for i in range(len(signal) - (AUTOCORRELATION_LENGTH - 1)):
        autocorrelations.append(getAutocorrelation(signal[i:i+AUTOCORRELATION_LENGTH]))   

    # # DEBUG: autocorrelations
    # positions = range(1080 - (AUTOCORRELATION_LENGTH - 1))
    # plt.plot(positions, autocorrelations)
    # plt.savefig(f'./byproduct/0/autocorrelations/autocorrelation-{idx+1}.jpg')
    # plt.clf()

    # find the position of bit 0 and bit 1 (positions of transition)
    # transition_position = get_transition_position(signal)
    min_positions = argrelextrema(np.array(autocorrelations), np.less)[0]
    # token_sequence = get_token_sequence(signal, transition_position, min_positions)
    token_sequence = get_token_sequence(signal, min_positions)
    token_sequences.append(token_sequence)

    # DEBUG: observe tokens
    with open(f'./byproduct/0/tokens/tokens-{idx+1}', 'w') as file:
      writer = csv.writer(file)
      writer.writerow(token_sequence)
  
  return token_sequences 

def getAutocorrelation(signal:list) -> int:
    sum = 0
    signal_length = len(signal)
    for diff in range(signal_length):
       for i in range(signal_length - diff):
           sum = sum + signal[i] * signal[i + diff]
    return sum

def get_token_sequence(signal:list, min_positions:list) -> list:
  # check whether there is double da?

  # check whether it's  bit_0 or bit_1 ?

  token_ids = []
  for pos in min_positions:
    r = -2
    token_id = -1
    s = signal[pos:pos + AUTOCORRELATION_LENGTH]
    for idx, token in enumerate(TOKEN_EXTEND):
      curr_r = np.corrcoef(s, token)[0, 1]
      if curr_r > r:
        r = curr_r
        token_id = idx
    token_ids.append(token_id)
  return token_ids

# issue: when facing das, there are multiple local minimums in the near area -> which should be counted as one double das
# issue: some delimiter looks like bit_0 or bit_1  in  a small scale
# issue: double da is weird on the graph of autocorrelation

# observation
# bit 0/1 will not appearl consecutively
# only delimiter da can appear consecutively
# a frame can contains at most 4 tokens  (eg: 1 da da 1)
# so if the result is   ? 3 ?   then ? = bit 0 or bit 1 
