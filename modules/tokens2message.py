from Packet import Packet

def tokens2message(tokens):
 
  packet = Packet()

  bit0, bit1, Da, Db, Fa, Fb = range(6)

  Da_in_last_token = False
  message_start = False

  for token in tokens:

    if Da_in_last_token:
      message_start = True

    if message_start:
      packet.putPattern(token)

    Da_in_last_token = (Da in token)

    if packet.messageFull():
      return packet.getMessage()

  return packet.getMessage()
  raise Exception("Some bits are totally lost.")


# DEBUG
if __name__ == '__main__':
  tokens = [[2,0,2], [5,1,3,1], [4,1,5,0]]
  print(tokens2message(tokens))