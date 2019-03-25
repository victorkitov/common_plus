def rm_from_list(lst,values):
    '''Remove a sequence of values (values) from list (lst). List is modified inplace.'''
    for value in values:
        lst.remove(value)
        
        
        
'''List concatenations'''

def concat_elementwise(base,addition):
    '''
    concat_elementwise([1,2,3],[11,12,13]) returns:
    [[1, 2, 3, 11], [1, 2, 3, 12], [1, 2, 3, 13]]
    '''
    return [base+[addition[i]] for i in range(len(addition))]


def concat_cumulative(base,addition):
    '''
    concat_cumulative([1,2,3],[11,12,13]) returns:
    [[1, 2, 3, 11], [1, 2, 3, 11, 12], [1, 2, 3, 11, 12, 13]]
    '''
    return [base+addition[:i] for i in range(1,len(addition)+1)]