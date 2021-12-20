class Packet:

    bit0, bit1, Da, Db, Fa, Fb, bitx = range(7)
    
    def __init__(self):
        self.bitcontent = [6] * 30
        self.lastPatternIndex = -1

    def __len__(self):
        return len(self.bitcontent)

    def __str__(self):
        return self.getMessage()

    def getMessage(self):
        return str(self.bitcontent).replace('[', '').replace(']', '').replace(', ','')

    def getTokens(self):
        
        tokens = []

        # Preamble
        tokens += [Packet.Da, Packet.bit0] *5
        tokens += [Packet.Da, Packet.Da, Packet.bit0]

        # Content
        for i in range(30):
            bit = self.bitcontent[i]
            tokens += [Packet.Db, bit, Packet.Fa, bit, Packet.Fb, bit]
        
        return tokens

    def matchIndex(self, target):

        def fit(string, pattern):

            L = len(string)
            
            for i in range(L):

                if string[i] == pattern[i]:
                    continue
                
                elif string[i] == Packet.bitx:
                    if pattern[i] in [Packet.bit0, Packet.bit1]:
                        continue
                else:
                    return False
            return True

        tokens = self.getTokens()
        L_ta = len(target)
        L_to = len(tokens)

        available_index = []

        for i in range(L_to-L_ta+1):

            substring = tokens[i:i+L_ta]
            if fit(substring, target):
                available_index.append(i)

        return available_index

    def putPattern(self, pattern):

        available_indices = self.matchIndex(pattern)
        if not available_indices:
            return
        
        goal = -1
        for index in available_indices:
            if index > self.lastPatternIndex:
                goal = index
                break

        if goal == -1:
            raise Exception("No suitable place to put this frame")
        

        self.lastPatternIndex = goal

        for diff, e in enumerate(pattern):
            if e in [Packet.bit0, Packet.bit1]:
                self.bitcontent[(goal-13+diff) // 6] = e

    def messageFull(self):

        for bit in self.bitcontent:
            if bit == Packet.bitx:
                return False
        return True

# DEBUG:  
if __name__ == '__main__':
    
    packet = Packet()
    
    print(packet.getTokens(), end='\n\n')
    packet.putPattern([5, 1, 3, 1])

    print(packet.getTokens(), end='\n\n')
    packet.putPattern([5, 1, 3, 1])

    print(packet.getTokens(), end='\n\n')
    packet.putPattern([5, 1, 3, 1])

    print(packet)

