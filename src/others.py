"""
Author: Ogia Wei
Date: 2025-07-05
Version: 1.0
Description: other scripts.
"""

################### other scripts ###################
##### When importing packages, avoid circular imports. #####
def display_info(content, length=80):
    """
    Display running information
    """
    num = int((length - len(content) - 2)/2)
    out = "{} {} {}".format("#"*num, content, "#"*num)
    out = out if len(out) == length else out+"#"
    # print(out)
    return out

def print_hash(length=80):
    """
    Print the # of the specified length 
    """
    print("#"*length)

