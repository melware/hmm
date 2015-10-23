#!/usr/bin/env python3

from HiddenMarkovModel import HMM

# measure program runtime
import time
time_start = time.time()

# train an HMM on the training file
with open("materials/POSData/training-full.pos") as f_train:
  hmm = HMM(f_train)

# read in test file and output a tagged corpus to test_out
with open("materials/POSData/test.text") as test_in:
  with open("materials/POSData/shannon-li-test.pos", 'w') as test_out:
    sentence = []
    for word in test_in:  # words are line-delimited
      if word != '\n':
        sentence.append(word[:-1])  # don't include the '\n' at the end of each word
      else:  # we have reached the end of a "sentence"
        # convert the words to lowercase before running viterbi on it 
        tag_sequence = hmm.run_viterbi([word.lower() for word in sentence])
        # write result into output file 
        for w in range(len(sentence)):
          line = sentence[w] + '\t' + tag_sequence[w] + '\n'
          test_out.write(line)
        test_out.write('\n')
        sentence = []
print("Finished. Total time elapsed:{} seconds".format(time.time()-time_start))


