"""
Code for compressing and decompressing using Huffman compression.
"""

from nodes import HuffmanNode, ReadNode


# ====================
# Helper functions for manipulating bytes


def get_bit(byte, bit_num):
    """ Return bit number bit_num from right in byte.

    @param int byte: a given byte
    @param int bit_num: a specific bit number within the byte
    @rtype: int

    >>> get_bit(0b00000101, 2)
    1
    >>> get_bit(0b00000101, 1)
    0
    """
    return (byte & (1 << bit_num)) >> bit_num


def byte_to_bits(byte):
    """ Return the representation of a byte as a string of bits.

    @param int byte: a given byte
    @rtype: str

    >>> byte_to_bits(14)
    '00001110'
    """
    return "".join([str(get_bit(byte, bit_num))
                    for bit_num in range(7, -1, -1)])


def bits_to_byte(bits):
    """ Return int represented by bits, padded on right.

    @param str bits: a string representation of some bits
    @rtype: int

    >>> bits_to_byte("00000101")
    5
    >>> bits_to_byte("101") == 0b10100000
    True
    """
    return sum([int(bits[pos]) << (7 - pos)
                for pos in range(len(bits))])


# ====================
# Functions for compression


def make_freq_dict(text):
    """ Return a dictionary that maps each byte in text to its frequency.

    @param bytes text: a bytes object
    @rtype: dict(int,int)

    >>> d = make_freq_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    freq = {}
    for i in range(len(text)):
        if text[i] not in freq:
            freq[text[i]]= 1
        else:
            freq[text[i]]+=1
    return freq

def dict_invert(freq_dict):
    '''inverts a freq_dict (number: frequency) -> (frequency: number)'''
    new = {}
    for letter in freq_dict:
        freq = freq_dict[letter]
        if not (freq in new):
            new[freq] = letter
        else:
            new[freq].append(letter)
    #print(new)
    return new


            
def huffman_tree(freq_dict):
    """ Return the root HuffmanNode of a Huffman tree corresponding
    to frequency dictionary freq_dict.

    @param dict(int,int) freq_dict: a frequency dictionary
    @rtype: HuffmanNode

    >>> freq = {2: 6, 3: 4}
    >>> t = huffman_tree(freq)
    >>> result1 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> result2 = HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))
    >>> t == result1 or t == result2
    True
    """
    stack = []
    for item in freq_dict:
        stack.append((freq_dict[item], HuffmanNode(item)))

    if len(stack) == 1:
        return HuffmanNode(None, stack[0][1], stack[0][1])

    while len(stack) > 1:
        stack = sorted(stack, key=lambda x: x[0], reverse=False)
        l = stack.pop(0)
        r = stack.pop(0)
        stack.append((l[0] + r[0], HuffmanNode(left=l[1], right=r[1])))
    return stack[0][1]


def get_codes(tree):
    """ Return a dict mapping symbols from Huffman tree to codes.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: dict(int,str)

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    return (help_get_codes(tree, '', {}))

def help_get_codes(tree, code, code_dict):
    if isinstance(tree.symbol, int):
        code_dict[tree.symbol] = code
        return code_dict
    else:
        help_get_codes(tree.left, code+ '0', code_dict)
        help_get_codes(tree.right, code+'1', code_dict)
    return code_dict
    

def number_nodes(tree):
    """ Number internal nodes in tree according to postorder traversal;
    start numbering at 0.

    @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    help_number_nodes(tree,0)
    


def help_number_nodes(tree, number):
    #Postorder Left, Right, Node
    if tree.symbol!= None or (tree.left ==None and tree.right == None):
        return number
    else:
        #go through left side of the tree checking for empty nodes
        if tree.left:
            number = help_number_nodes(tree.left, number)
        #checks the right side for nodes
        if tree.right:
            number = help_number_nodes(tree.right, number)
        #assigns the tree the largest number from the 2 sides
        tree.number = number
        return number+1
    

def avg_length(tree, freq_dict):
    """ Return the number of bits per symbol required to compress text
    made of the symbols and frequencies in freq_dict, using the Huffman tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: float

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(9)
    >>> tree = HuffmanNode(None, left, right)
    >>> avg_length(tree, freq)
    1.9
    """
    #Get the codes
    codes = get_codes(tree)
    
    #get the total number of bits needed by multiplying the freq with the code length
    total_bits = 0
    for keys in freq_dict:
        total_bits += freq_dict[keys]* len(codes[keys])
        
    #get average length (divide total bits by the total freqency)
    total_freq = sum(freq_dict.values())
    return (total_bits/total_freq)
        
def generate_compressed(text, codes):
    """ Return compressed form of text, using mapping in codes for each symbol.

    @param bytes text: a bytes object
    @param dict(int,str) codes: mapping from symbols to codes
    @rtype: bytes

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    final = []
    t = ''
    for item in text:
        t += codes[item]
        if len(t) > 8:
            final += [bits_to_byte(t[:8])]
            t = t[8:]
    if t:
        final += [bits_to_byte(t)]
    return bytes(final)

def tree_to_bytes(tree):
    """ Return a bytes representation of the Huffman tree rooted at tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes

    The representation should be based on the postorder traversal of tree
    internal nodes, starting from 0.
    Precondition: tree has its nodes numbered.

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    """
    

    lis = []
    def detect(tree):
        '''
        @param HuffmanNode tree
        @rtype: Nonetype
        '''
        if (tree.left is not None) or (tree.right is not None):
            
        
            if tree.left.is_leaf():
                lis.append(0)
                lis.append(tree.left.symbol)
            if not tree.left.is_leaf():
                lis.append(1)
                lis.append(tree.left.number)

            if tree.right.is_leaf():
                lis.append(0)
                lis.append(tree.right.symbol)
            if not tree.right.is_leaf():
                lis.append(1)
                lis.append(tree.right.number)


    def postorder_travel(node):
        '''
        @param HuffmanNode node
        @rtype: bytes
        '''
        if node is not None:
            if not node.symbol:
                return postorder_travel(node.left), \
                       postorder_travel(node.right), detect(node)

    postorder_travel(tree)
    return bytes(lis)


def num_nodes_to_bytes(tree):
    """ Return number of nodes required to represent tree (the root of a
    numbered Huffman tree).

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes
    """
    return bytes([tree.number + 1])


def size_to_bytes(size):
    """ Return the size as a bytes object.

    @param int size: a 32-bit integer to convert to bytes
    @rtype: bytes

    >>> list(size_to_bytes(300))
    [44, 1, 0, 0]
    """
    # little-endian representation of 32-bit (4-byte)
    # int size
    return size.to_bytes(4, "little")


def compress(in_file, out_file):
    """ Compress contents of in_file and store results in out_file.

    @param str in_file: input file to compress
    @param str out_file: output file to store compressed result
    @rtype: NoneType
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = make_freq_dict(text)
    tree = huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (num_nodes_to_bytes(tree) + tree_to_bytes(tree) +
              size_to_bytes(len(text)))
    result += generate_compressed(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

#first byte is 0 if the left subtree is a leaf, else its 1
# second byte is the symbol of the left subtree else node number of the tree
# Third byte is 0 if the right subtree is a leaf, else its 1
# Fourth byte is the symbol of the right subtree if n is a leaf, else node number of the right tree

def generate_tree_general(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes nothing about the order of the nodes in node_lst.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in 'node_lst'
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(10, None, None), \
HuffmanNode(12, None, None)), \
HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(7, None, None)))
    """
    # todo
    #find root node
    root = node_lst[root_index]
    
    #Checking through to see if the left subtree is a node or a leaf
    if root.l_type == 0:
        #since its a leaf, store the 'node' version of it
        left = HuffmanNode(root.l_data)
    else:
        #since it is a node, go deeper into the node
        #let the new index be the value of the node from the root
        #Ex, root = (1,1,1,0) Left tree's node is 1 <- pass this value as the index
        left = generate_tree_general(node_lst, root.l_data)
    
    #same bus with the right side
    if root.r_type == 0:
        right = HuffmanNode(root.r_data)
    else:
        right = generate_tree_general(node_lst, root.r_data)
    
    #Make a HuffmanNode version
    return HuffmanNode(None, left, right)    

def generate_tree_postorder(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes that node_lst represents a tree in postorder.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in 'node_lst'
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(5, None, None), \
HuffmanNode(7, None, None)), \
HuffmanNode(None, HuffmanNode(10, None, None), HuffmanNode(12, None, None)))
    """
    # todo
    return post0rder3(node_lst, root_index)

def post0rder3(node_lst, root_index):
   
    root = node_lst[root_index]
    tree = HuffmanNode(None, None, None)
   
 
    if root.r_type == 0:
        tree.right = HuffmanNode(root.r_data)
        num = 0
    else:
        tree.right = post0rder3(node_lst, root_index - 1)
        #print(tree.right)
        num = help_post(tree.right)
           
    if root.l_type == 0:        
        tree.left = HuffmanNode(root.l_data)
    else:       
        tree.left = post0rder3(node_lst, root_index - num - 1)
 
    return tree    


def generate_uncompressed(tree, text, size):
    """ Use Huffman tree to decompress size bytes from text.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @param bytes text: text to decompress
    @param int size: number of bytes to decompress from text.
    @rtype: bytes
    """
    # todo
    
    def rever(d):
        '''
        @param dict d
        @rtype: dict
        '''
        final = {}
        for key, value in d.items():
            final[value] = key
        return final    


    codes = rever(get_codes(tree))

    bits = ''
    for char in list(text):
        bits += byte_to_bits(char)

    uncompressed = []
    txt = ''
    for bit in bits:
        txt += bit
        if txt in codes:
            uncompressed.append(codes[txt])
            txt = ''
            if len(uncompressed) == size:
                break

    return bytes(uncompressed)
    
    
    


def bytes_to_nodes(buf):
    """ Return a list of ReadNodes corresponding to the bytes in buf.

    @param bytes buf: a bytes object
    @rtype: list[ReadNode]

    >>> bytes_to_nodes(bytes([0, 1, 0, 2]))
    [ReadNode(0, 1, 0, 2)]
    """
    lst = []
    for i in range(0, len(buf), 4):
        l_type = buf[i]
        l_data = buf[i+1]
        r_type = buf[i+2]
        r_data = buf[i+3]
        lst.append(ReadNode(l_type, l_data, r_type, r_data))
    return lst


def bytes_to_size(buf):
    """ Return the size corresponding to the
    given 4-byte little-endian representation.

    @param bytes buf: a bytes object
    @rtype: int

    >>> bytes_to_size(bytes([44, 1, 0, 0]))
    300
    """
    return int.from_bytes(buf, "little")


def uncompress(in_file, out_file):
    """ Uncompress contents of in_file and store results in out_file.

    @param str in_file: input file to uncompress
    @param str out_file: output file that will hold the uncompressed results
    @rtype: NoneType
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_size(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(generate_uncompressed(tree, text, size))


# ====================
# Other functions

def improve_tree(tree, freq_dict):
    """ Improve the tree as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to freq_dict.

    @param HuffmanNode tree: Huffman tree rooted at 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(99), HuffmanNode(100))
    >>> right = HuffmanNode(None, HuffmanNode(101), \
    HuffmanNode(None, HuffmanNode(97), HuffmanNode(98)))
    >>> tree = HuffmanNode(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    # todo
    tups = [(byte, freq) for byte, freq in freq_dict.items()]
    tups.sort(key=lambda x: x[1])

    traversal = [tree]
    
    while traversal:
        node = traversal.pop(0)
        if node.is_leaf():
            node.symbol = tups.pop()[0]
        for n in [node.left, node.right]:
            if n != None:
                traversal.append(n)    

if __name__ == "__main__":
    # TODO: Uncomment these when you have implemented all the functions
    # import doctest
    # doctest.testmod()

    import time

    mode = input("Press c to compress or u to uncompress: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress(fname, fname + ".huf")
        print("compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "u":
        fname = input("File to uncompress: ")
        start = time.time()
        uncompress(fname, fname + ".orig")
        print("uncompressed {} in {} seconds."
              .format(fname, time.time() - start))
