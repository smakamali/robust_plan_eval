import numpy as np
import torch
import xml.etree.ElementTree as ET

class TreeNode():
    def _init_(self):
        self.val = None
        self.leftChild = None
        self.rightChild = None
    def print(self):
        print(self.val.tag,self.val.attrib)
        if self.leftChild is not None:
            print('left')
            self.leftChild.print()
        if self.rightChild is not None:
            print('right')
            self.rightChild.print()

class TreeConvolutionError(Exception):
    pass


from sklearn.preprocessing import OneHotEncoder

def featurizetree(root,id_tab):

    def to_2d(list):
        return np.array(list).reshape(-1,1)


    operators = to_2d([
        'HSJOIN', '^HSJOIN', 'HSJOIN^',
        'NLJOIN', '^NLJOIN','NLJOIN^',
        'MSJOIN','^MSJOIN','MSJOIN^',
        'TBSCAN','FILTER','SORT',
        'IXSCAN','FETCH','LPREFETCH','GRPBY'
        ])
    tabs = to_2d([tab.split('.')[1] for tab in id_tab.keys()])
    
    op_enc = OneHotEncoder(handle_unknown='ignore')
    op_enc.fit(operators)

    tab_enc = OneHotEncoder(handle_unknown='ignore')
    tab_enc.fit(tabs)

    other_op = op_enc.transform(to_2d(['Opx'])).toarray().reshape(-1)
    other_tab = tab_enc.transform(to_2d(['Qx'])).toarray().reshape(-1)

    num_tabs = tabs.shape[0]

    # capture onehot encoding for each access operator
    for node in root.iter():
        opoh = op_enc.transform(to_2d([node.tag])).toarray().reshape(-1)

        tab_detected = False
        for attrib in ['TABID','TABLE']:
            if attrib in node.attrib.keys():
                taboh = tab_enc.transform(to_2d([node.attrib[attrib]])).toarray().reshape(-1)
                tab_detected = True
                break

        if not tab_detected:
            taboh = other_tab
        
        stats=[]
        for attrib in ['TOTAL_COST','OUTPUT_CARD','SELECTIVITY']:
            if attrib in node.attrib.keys():
                stats.append(float(node.attrib[attrib]))
            else:
                stats.append(0)
        
        node.set('onehot',np.concatenate((stats,opoh,taboh)))
    
    # capture underlying tables for non-access operators
    for node in root.iter():
        if 'TABID' not in node.attrib.keys() or 'TABLE' not in node.attrib.keys() :
            childrenSum = other_tab
            for elem in node.iter():
                childrenSum = np.logical_or(childrenSum,np.array(elem.attrib['onehot'][-num_tabs:]))
            node.attrib['onehot'][-num_tabs:] = childrenSum
            
    return root

def getXMLChildren(xmlnode):
    children = []
    null = ET.Element("Null")
    for child in xmlnode:
        children.append(child)
    if len(children) == 1:
        return children[0], null
    elif len(children) == 2:
        return children[0], children[1]
    else:
        return None, None

def xmltotree(xmlroot):
    if not isinstance(xmlroot, ET.Element):
        return None
    else:
        tre = TreeNode()
        tre.val = xmlroot
        if xmlroot.tag !="Null":
            left,right = getXMLChildren(xmlroot)
            tre.leftChild = xmltotree(left)
            tre.rightChild = xmltotree(right)
        else:
            tre.leftChild = None
            tre.rightChild = None
        return tre


def _is_leaf(x, left_child, right_child):
    has_left = left_child(x) is not None
    has_right = right_child(x) is not None
    
    if has_left != has_right:
        raise TreeConvolutionError(
            "All nodes must have both a left and a right child or no children"
        )

    return not has_left

def _flatten(root, transformer, left_child, right_child):
    """ turns a tree into a flattened vector, preorder """
    
    
    if not callable(transformer):
        raise TreeConvolutionError(
            "Transformer must be a function mapping a tree node to a vector"
        )
    
    if not callable(left_child) or not callable(right_child):
        raise TreeConvolutionError(
            "left_child and right_child must be a function mapping a "
            + "tree node to its child, or None"
        )
    
        
    accum = []
    

    def recurse(x):
        
        if _is_leaf(x, left_child, right_child):
            accum.append(transformer(x))
            return
        
        
        accum.append(transformer(x))
        
        recurse(left_child(x))
        
        recurse(right_child(x))
        

    recurse(root)

    try:
        accum = [np.zeros(accum[0].shape)] + accum
    except:
        raise TreeConvolutionError(
            "Output of transformer must have a .shape (e.g., numpy array)"
        )
    
    return np.array(accum)

def _preorder_indexes(root, left_child, right_child, idx=1):
    """ transforms a tree into a tree of preorder indexes """
    
    if not callable(left_child) or not callable(right_child):
        raise TreeConvolutionError(
            "left_child and right_child must be a function mapping a " +
            "tree node to its child, or None"
        )


    if _is_leaf(root, left_child, right_child):
        # leaf
        return idx

    def rightmost(tree):
        if isinstance(tree, tuple):
            return rightmost(tree[2])
        return tree
    
    left_subtree = _preorder_indexes(left_child(root), left_child, right_child,
                                     idx=idx+1)
    
    max_index_in_left = rightmost(left_subtree)
    right_subtree = _preorder_indexes(right_child(root), left_child, right_child,
                                      idx=max_index_in_left + 1)

    return (idx, left_subtree, right_subtree)
    
def _tree_conv_indexes(root, left_child, right_child):
    """ 
    Create indexes that, when used as indexes into the output of `flatten`,
    create an array such that a stride-3 1D convolution is the same as a
    tree convolution.
    """
    
    if not callable(left_child) or not callable(right_child):
        raise TreeConvolutionError(
            "left_child and right_child must be a function mapping a "
            + "tree node to its child, or None"
        )
    
    index_tree = _preorder_indexes(root, left_child, right_child)

    def recurse(root):
        if isinstance(root, tuple):
            my_id = root[0]
            left_id = root[1][0] if isinstance(root[1], tuple) else root[1]
            right_id = root[2][0] if isinstance(root[2], tuple) else root[2]
            yield [my_id, left_id, right_id]
                                           
            yield from recurse(root[1])
            yield from recurse(root[2])
        else:
            yield [root, 0, 0]

    return np.array(list(recurse(index_tree))).flatten().reshape(-1, 1)

def _pad_and_combine(x):
    assert len(x) >= 1
    assert len(x[0].shape) == 2

    for itm in x:
        if itm.dtype == np.dtype("object"):
            raise TreeConvolutionError(
                "Transformer outputs could not be unified into an array. "
                + "Are they all the same size?"
            )
    
    second_dim = x[0].shape[1]
    for itm in x[1:]:
        assert itm.shape[1] == second_dim

    max_first_dim = max(arr.shape[0] for arr in x)

    vecs = []
    for arr in x:
        padded = np.zeros((max_first_dim, second_dim))
        padded[0:arr.shape[0]] = arr
        vecs.append(padded)

    return np.array(vecs)

# Takes TreeNode class as input and extracts onehot encoding as numpy array
def transformer(node):
    return np.array(node.val.attrib['onehot'])
# Takes TreeNode class as input and extracts leftChild
def left_child(node):
    return node.leftChild
# Takes TreeNode class as input and extracts rightChild
def right_child(node):
    return node.rightChild

def prepare_trees(trees, transformer, left_child, right_child, cuda=False):
    
    flat_trees = [_flatten(x, transformer, left_child, right_child) for x in trees]
    
    
    flat_trees = _pad_and_combine(flat_trees)
    
    flat_trees = torch.Tensor(flat_trees)
    

    
    flat_trees = flat_trees.transpose(1, 2)
    if cuda:
        flat_trees = flat_trees.cuda()

    indexes = [_tree_conv_indexes(x, left_child, right_child) for x in trees]
    indexes = _pad_and_combine(indexes)
    indexes = torch.Tensor(indexes).long()

    if cuda:
        indexes = indexes.cuda()

    return (flat_trees, indexes)
                    
