class HMM_Table:
  def __init__(self):
    """
    a 2x2 matrix for a probability table in an HMM
    :return:
    """
    self.matrix = {}
    self.is_prob = False  # are the values in this matrix probabilities? (not frequency counts)


  def get(self, pos, word, word_list):
    if word not in word_list:
      # print("could not find word:{} as pos:{}".format(word, pos))
      # TODO better way of tagging unknown words
      self.matrix[pos][word] = min(self.matrix[pos].values())

    if word not in self.matrix[pos].keys():
      return 0
    return self.matrix[pos][word]

  def add(self, x, y):
    """

    :param x:
    :param y:
    """
    if x not in self.matrix:
      self.matrix[x] = {}
    if y not in self.matrix[x]:
      self.matrix[x][y] = 0
    self.matrix[x][y] += 1

  def to_probability(self):
    """
    convert values in this table from frequency counts to probabilities
    (this function is idempotent!)
    """

    if not self.is_prob:
      for x in self.matrix.keys():
        x_freq = 0
        for val in self.matrix[x].values():
          x_freq += val
        for y in self.matrix[x].keys():
          self.matrix[x][y] /= x_freq

      self.is_prob = True

  def to_probability_table(self, states):
    """
    :param states = lookup table pos_tag->index
    """
    if not self.is_prob:
      temp_table = [[0 for y in states] for x in states]
      for prev in self.matrix.keys():
        total_freq = 0
        for val in self.matrix[prev].values():
          total_freq += val
        for pos in self.matrix[prev].keys():
          temp_table[states[prev]][states[pos]] = self.matrix[prev][pos] / total_freq

      self.is_prob = True
      return temp_table

  def output_to_file(self, output_file):
    """
      writes this table out to a file
      :param output_file: file to output to
    """
    for x in self.matrix.keys():
      for y in self.matrix[x].keys():
        output_file.write(x + ', ' + y + ', ' + str(self.matrix[x][y]) + '\n')

START_TAG = 'START'
END_TAG = 'END'


class HMM:
  def __init__(self, training_file):
    self.transition = [[]]
    self.likelihood = HMM_Table()  # stays a HMM_Table

    self.states = []
    self.states_index = {}
    self.words = []
    self.words_index = {}

    self.train(training_file)

  def add_state(self, state):
    self.states_index[state] = len(self.states)
    self.states.append(state)

  def add_word(self, word):
    self.words_index[word] = len(self.words)
    self.words.append(word)

  def train(self, training_file):
    """
    Trains itself on a corpus
    Creates the likelihood and transition matrices
    :param training_file: file containing a corpus
            format: line-delimited (word,tag) pairs.
                    (word,tag) pairs are tab-delimited
    """
    temp_transition = HMM_Table()
    prev_pos = ''

    self.add_state(START_TAG)
    for line in training_file:
      line = line.split('\t')
      word = line[0].lower()
      pos = line[1][:-1]  # strip the newline character

      if word != '':  # don't count blanks (sentence start/stop) in the likelihood table
        if word not in self.words_index:
          self.add_word(word)
        self.likelihood.add(pos, word)

      if pos != '' and pos not in self.states_index:
        self.add_state(pos)

      # add to transition table
      if pos == '':
        temp_transition.add(prev_pos, END_TAG)
      elif prev_pos == '':
        temp_transition.add(START_TAG, pos)
      else:
        temp_transition.add(prev_pos, pos)

      prev_pos = pos

    self.add_state(END_TAG)
    self.transition = temp_transition.to_probability_table(self.states_index)
    self.likelihood.to_probability()

  def run_viterbi(self, input_sentence):
    """
    Computes the most-likely tag sequence for the given sentence
    :param sentence: a list of words
    :returns a list of POS tags, corresponding to each word in the sentence
    """
    if len(input_sentence) < 1: return ['']
    sentence = [''] + input_sentence + ['']
    # initialize (POS x sentence) viterbi and backpointer tables
    viterbi = [[0 for w in sentence] for p in self.states]
    backpointer = [[-1 for w in sentence] for p in self.states]

    # set starting values 
    viterbi[0][0] = 1
    s = self.states_index[START_TAG]
    for p in range(1, len(viterbi)-1):
      viterbi[p][1] = self.transition[0][p] * self.likelihood.get(self.states[p], sentence[1], self.words)
      backpointer[p][1] = s 

    # populate the viterbi and backpointer tables
    for word in range(2, len(sentence)-1):
      # print("testing for {}. word#{}:{}".format(word,w,self.words[w]))
      for p in range(1, len(viterbi)-1):
        cur_likelihood = self.likelihood.get(self.states[p], sentence[word], self.words)
        # print("p:{}; w:{}; likelihood:{}".format(self.states[p], self.words[w], cur_likelihood))
        max_result = 0
        max_prev = 1
        for prev in range(1, len(viterbi)-1):
          result = viterbi[prev][word-1] * self.transition[prev][p] * cur_likelihood
          if result > max_result:
            max_result = result
            max_prev = prev
        # print("inserting {} into viterbi[{}][{}]".format(max_result,p,word))
        backpointer[p][word] = max_prev
        viterbi[p][word] = max_result

    max_result = 0
    last_pos = 1
    lw = len(sentence)-2
    e = len(self.states)-1  # end tag
    # fill in the final state (end tag)
    for p in range(1, len(viterbi)-1):
      result = viterbi[p][lw] * self.transition[p][e]
      if result > max_result:
        max_result = result
        last_pos = p
    # print("last_pos for {} = {}".format(sentence[lw], self.states[last_pos]))
    backpointer[e][lw+1] = last_pos
    viterbi[e][lw+1] = max_result

    tag_seq = [self.states[last_pos]]
    for w in range(lw, 1, -1):  # from last word to first word
      # print("w:{}; last_pos:{}".format(w,last_pos))
      last_pos = backpointer[last_pos][w]
      tag_seq.insert(0, self.states[last_pos])
    
    return tag_seq
#
#  def print_backpointer(self, backpointer):
#    for i in range(len(backpointer)):
#      print(self.states[i] + ':\t\t', end="")
#      for j in range(len(backpointer[i])):
#        print('{}\t'.format(self.states[backpointer[i][j]]), end="")
#      print()
#
#    print()
#
#
#  def print_viterbi(self, viterbi):
#    for i in range(len(viterbi)):
#      print(self.states[i] + ':\t\t', end="")
#      for j in range(len(viterbi[i])):
#        print('{}\t'.format(viterbi[i][j]), end="")
#      print()
#
#    print()
#

      
    




#
#    # calculate the most likely tag for the last word
#    lw = len(sentence)-1  # index of the last word
#
#    last_tag = max(viterbi[lw].keys(), key=(lambda k: viterbi[lw][k]*self.transition.get(k, END_TAG)))
#
#    tag_sequence = [last_tag]
#    # follow backpointer to output a most likely tag sequence
#    for w in range(len(sentence)-1, 0, -1):
#      last_tag = backpointer[w][last_tag]
#      # print(last_tag+' ')
#      tag_sequence.insert(0, last_tag)
#
#    # print('\n')
#    # print("tag:{}\nsentence:{}".format(tag_sequence, sentence))
#    return tag_sequence
  
