# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 14:15:17 2017

@author: david
"""  

import os
import glob
import numpy as np
import heapq
import itertools
from collections import defaultdict
import argparse
try:
    from rich import print
except ImportError:
    ...
import multiprocessing as mp
try: 
    import queue
except ImportError:
    import Queue as queue

from ..phylo import tree as tree_lib
from ..utils import parallel_task_manager
from .rooting import store_species_sets
    
def detach_and_cleanup(top, n):
    """
    top - the node that all the nodes in the current analysis will be below
    nUp - the node above n originally, from here to nTop needs the cached data updated
    """
    n_up = n.up
    n = n.detach()
    ch = n_up.get_children()
    if len(ch) == 1:
        if n_up.is_root():
            # can't delete the root, need to detach the child nodes and reattach them to the root
            for c in ch[0].get_children():
                n_up.add_child(c.detach())
            ch[0].delete()
            return n, n_up, top
        else:
            if top == n_up: top = n_up.up 
            n_up_new = n_up.up
            n_up.delete()
            return n, n_up_new, top
    return n, n_up, top

def graft_and_update(top, n, s):
    """
    Remove n from its location in the tree and make it sister to s. Update only the sp_down data. NOT the sp_up data.
    Args:
        nTop - node being reconcilled
        n - node to move
        s - node to make n sister to.
    Returns:
        The root of the tree
    Implementation:
        n and s could be anywhere in relation to one another, need to change all the species sets that could be affected
    """
    n, n_up, top = detach_and_cleanup(top, n)
    parent = s.up
    if parent == None:
        new = tree_lib.TreeNode()
        new.add_feature("sp_up", set())
        parent = new
        top = new
        new.add_child(n)
        s = s.detach()
        new.add_child(s)
        s.sp_up = n.sp_down
    else:
        new = parent.add_child()
        new.add_child(n)
        s = s.detach()
        new.add_child(s)
    
    # sort out distances, new node will have a default distance of 1.0, correct this
    new.dist = 0.1*s.dist
    s.dist   = 0.9*s.dist
    
    # sort out sp_down data - can do a more efficient routine later if necessary (updating only what needs updating). Otherwise:
    # all nodes on path from nTop to s need sp_down updating
    new.add_feature("sp_down", s.sp_down.union(n.sp_down))
    parent.sp_down = parent.sp_down.union(n.sp_down)
    if not parent == top:
        r = parent.up
        while r != top:
            r.sp_down = set.union(*[ch.sp_down for ch in r.get_children()])
            r = r.up
    # all nodes on path from nTop to n need sp_down updating
    r = n_up
    if r != None:
        while r != top:
            r.sp_down = set.union(*[ch.sp_down for ch in r.get_children()])
            r = r.up
    return new.get_tree_root()
    
# def GraftTripartAndUpdate(nTop, s1, s2, p):
#     """
#     Make s1 and s2 sister clades and hang the from parent node p
#     Args:
#         s1, s2 - nodes to be made sisters of each other
#         p - node to hang (s1,s2) from
#     """
#     d1 = s1.up.dist
#     s1, s1Parent , nTop = detach_and_cleanup(nTop, s1)   
#     d2 = s2.up.dist
#     s2, s2Parent, nTop = detach_and_cleanup(nTop, s2)   
#     new = p.add_child()
#     new.dist = 0.5*(d1+d2)
#     new.add_child(s1)
#     new.add_child(s2)
#     new.add_feature("sp_down", s1.sp_down.union(s2.sp_down))
#     n = s1Parent
#     while n != nTop:
#         n.sp_down = set.union(*[ch.sp_down for ch in n.get_children()])
#         n = n.up
#     n = s2Parent
#     while n != nTop:
#         n.sp_down = set.union(*[ch.sp_down for ch in n.get_children()])
#         n = n.up
#     nTop.sp_down = set.union(*[ch.sp_down for ch in nTop.get_children()])
#     return new.get_tree_root()
        
def contains_monophyletic(e, O, i, I):
    """
    Check if set of species O is found only in a clade below e. If so, returns (True, n) where n is the clade otherwise (False, None)
    Args:
        e - the node to check
        O - the species set to look for
        i - the current iteration 
        I - the one-past-the-last iteration. Number of checks will be M-m
    Returns:
        qSuccess, node, depth
        where:
            qSuccess - True if found species set O to be monophyletic
            node - node of the monophyletic clade if qSuccess else the node it made it down to
            depth - depth of the node, 0=(e is O), 1=(it's child is O), etc.
    """
    E = e.sp_down
    if E == O:
        return True, e, i     
    i+=1
    if i == I or e.is_leaf(): return False, e, i-1
    ch = e.get_children()
    in_child = [bool(O & c.sp_down) for c in ch]
    if sum(in_child) > 1: return False, e, i-1         # bipart : cheange to if sum(inChild) == len(ch): return False, None 
    return contains_monophyletic(ch[in_child.index(True)], O, i, I)  # bipart need to check more than one potentially

def check_monophyly(node, taxa):
    """ This should be used as a wrapper of the ete method since that method returns false for a single node. It should return true
    Args:
        node - the node under which to search
        taxa - the list of taxa
    Returns
        bool - are the taxa monophyletic
    """
    if len(taxa) == 1:
        return (list(taxa)[0] in node)
    else: 
        return node.check_monophyly(target_attr='name', values=taxa)[0]

t = True
f = False

# dA = 1, dB = 0
# Subcases (S&O, S&U)
c10_a = {(f,f), (t,f), (t,t)}      
c10_b = {(f,t),}

# dA = 1, dB = 1
# Subcases (S&O, S&W, S&U)
c11_a = {(f,f,f), (t,t,t), (t,t,f), (t,f,t)}
c11_b = {(f,t,f), (f,f,t)}
c11_c = {(f,t,t),}
 
# dA = 2, dB = 0
# Subcases (S&V, S&Y, S&O, V&Y)
#c20_a = {(f,f,f,f),}
#c20_b = {(f,f,f,t), (f,f,t,f), (f,f,t,t), (f,t,f,t), (f,t,t,t), (t,f,t,t), (t,t,f,f), (t,t,f,t), (t,t,t,t)}
c20_a = {(f,f,f,f),(f,f,t,f),}
c20_b = {(f,f,f,t), (f,f,t,t), (f,t,f,t), (f,t,t,t), (t,f,t,t), (t,t,f,f), (t,t,f,t), (t,t,t,t)}
c20_c = {(f,t,f,f), (f,t,t,f), (t,f,f,f), (t,f,f,t), (t,f,t,f)}

def resolve(n, M):
    """This is an improved but also more staightforward implementation of resolve. It also considers far more sub-cases.
    It comes from writing up and generalising the first version.  
    
    Implementation:
        lower case letters are nodes, uppercase letters are species sets
    """
    if "recon" in n.features:
        sis = n.recon
        n.del_feature('recon')
        # cannot assume that sis[0] or sis[1] this will be just a single node 
        if check_monophyly(n, sis[0]) and check_monophyly(n, sis[1]):
            s0 = n.get_common_ancestor(sis[0]) if len(sis[0]) > 1 else (n&sis[0][0])
            s1 = n.get_common_ancestor(sis[1]) if len(sis[1]) > 1 else (n&sis[1][0])
            return graft_and_update(n, s0, s1)
    elif "recon_2" in n.features:
        """ (dest, target, move_out)
        move target to be sister to destination, move move_out to where target was
        Note, move_out is a list of lists, corresponding to potentially multiple nodes"""
        moves = n.recon_2
        n.del_feature('recon_2')
        dest, target, move_out = moves
        if check_monophyly(n, dest) and check_monophyly(n, target) and all([check_monophyly(n, m) for m in move_out]):
            # make all the move_out hang from a single node
            if len(move_out) == 1:
                sm_new = n.get_common_ancestor(move_out[0]) if len(move_out[0]) > 1 else (n&move_out[0][0])
            else:
                sm = n.get_common_ancestor(move_out[0]) if len(move_out[0]) > 1 else (n&move_out[0][0])
                sm_new = sm.up.add_child()
                dist = sm.up.dist
                sm_new.dist = dist
                sm_new_sp_down = set()
                for m in move_out:
                    sm = n.get_common_ancestor(m) if len(m) > 1 else (n&m[0])
                    sm_new_sp_down.update(sm.sp_down)
                    sm = sm.detach()
                    sm_new.add_child(sm)
                sm_new.add_feature("sp_down", sm_new_sp_down)
            s0 = n.get_common_ancestor(dest) if len(dest) > 1 else (n&dest[0])
            s1 = n.get_common_ancestor(target) if len(target) > 1 else (n&target[0])
            d = s1.get_sisters()[0].dist
            tree = graft_and_update(n, s1, s0)
            tree.dist = d
            s3 = tree.get_common_ancestor(dest + target)
            sister = s3.up.up
            top = sister.up if sister.up != None else sister
            return graft_and_update(top, sm_new, sister)
        
    ch = n.get_children()
    if len(ch) != 2: return n.get_tree_root()    #bipart
    X,Y = [c.sp_down for c in ch]
    O = X&Y
    if not O: return n.get_tree_root()# no overlap, nothing to do
    if X == Y: return n.get_tree_root()  # identical species sets, they're paralogues
    sis = n.get_sisters()
    s = sis[0] if len(sis) == 1 else None
    S = set() if (n.is_root() or len(n.get_sisters()) > 1) else set.union(*[nother.sp_down for nother in n.up.get_children() if nother != n]) # works for non-binary too
    
    success_a, n_a, d_a = contains_monophyletic(ch[0], O, 0, 3)   
    success_b, n_b, d_b = contains_monophyletic(ch[1], O, 0, 3)   
    if d_a+d_b > 2:
        if d_a == 2:
            success_a, n_a, d_a = contains_monophyletic(ch[0], O, 0, 2)   
        if d_b == 2:
            success_b, n_b, d_b = contains_monophyletic(ch[1], O, 0, 2)                  
        
    if (d_a==0 and d_b==1) or (d_a==1 and d_b==0): 
        # Case A
        # a is the one that branches
        # v is the child of A that overlaps with O
        if d_a == 1:
            a, b = ch
            v = n_a
        else:
            b, a = ch
            v = n_b
        U = set.union(*[c.sp_down for c in a.get_children() if c != v]) # allows for non-binary
        case = (bool(S&O), bool(S&U))  
        if case in c10_a:
            return graft_and_update(n, b, v)
        elif case in c10_b:
            n.up.add_feature("recon", (b.get_leaf_names(),s.get_leaf_names())) 
            return n.get_tree_root()
        return n.get_tree_root()
    elif (d_a==1 and d_b==1): 
        # Case B
        # v is the child of A that overlaps with O (ws are the others)
        # x is the child of B that overlaps with O
        v = n_a
        x = n_b
        cha = ch[0].get_children()
        chb = ch[1].get_children()
        U = set.union(*[c.sp_down for c in cha if c != v]) # allows for non-binary
        W = set.union(*[c.sp_down for c in chb if c != x]) # allows for non-binary
        case = (bool(S&O), bool(S&W), bool(S&U))
        if case in c11_a:
            return graft_and_update(n, x, v)
        elif case in c11_b:
            if W&S:
                # switch, so that ws, x are below a
                ws = [c for c in cha if c != v]
                w_leaves = [c.get_leaf_names() for c in ws]   # need to get the list of leaves that aren't in the node where the overlap is (for w.get_leaf_names(), below) AND need to update recon2 so that it will deal with multiple nodes in the ones it has to move
                x = v   
            else:
                ws = [c for c in chb if c != x]
                w_leaves = [c.get_leaf_names() for c in ws]
            n.up.add_feature("recon_2", (x.get_leaf_names(),s.get_leaf_names(),w_leaves)) 
            return n.get_tree_root()
        elif case in c11_c:
            ws = [c for c in chb if c != x]
            w_leaves = [g for c in ws for g in c.get_leaf_names()]
            n.up.add_feature("recon", (s.get_leaf_names(), w_leaves))  
            return n.get_tree_root()
        return n.get_tree_root()
    elif (d_a==2 and d_b==0) or (d_a==0 and d_b==2): 
        # Case C
        if d_a == 2:
            a,b = ch
        else:
            b,a = ch
        cha = a.get_children()
        i = next(ii for ii,c in enumerate(cha) if (c.sp_down & O))  # there will be only one node with an overlap
        u = cha[i]
        vs = [c for ii, c in enumerate(cha) if ii != i]
        V = set.union(*[c.sp_down for c in vs])
        chu = u.get_children()  
        i = next(ii for ii,c in enumerate(chu) if (c.sp_down & O))
        z = chu[i]
        ys = [c for ii, c in enumerate(chu) if ii != i]
        Y = set.union(*[c.sp_down for c in ys])
        case = (bool(S&V), bool(S&Y), bool(S&O), bool(V&Y))
        if case in c20_a:
            return graft_and_update(n, b, z)
        elif case in c20_b:
            # need to make v a single node
            if len(vs) == 1:
                vs_new = vs[0]
            else:
                vs_up = vs[0].up
                vs_new = vs_up.add_child()
                vs_new.dist = vs_up.dist
                vs_new_sp_down = set()
                for v in vs:
                    vs_new_sp_down.update(v.sp_down)
                    v = v.detach()
                    vs_new.add_child(v)
                vs_new.add_feature("sp_down", vs_new_sp_down)        
            return graft_and_update(n, vs_new, b)
        elif case in c20_c:
            n.up.add_feature("recon", (b.get_leaf_names(), s.get_leaf_names()))     
            return n.get_tree_root()
        return n.get_tree_root()
    return n.get_tree_root()

# def SpeciesOverlapDuplications(tree, GeneToSpecies):
#     species = list(set(map(GeneToSpecies, tree.get_leaf_names())))
#     genes = tree.get_leaf_names()
#     gDict = {gene:i for i,gene in enumerate(genes)}
#     sDict = {sp:i for i,sp in enumerate(species)}
#     for n in tree.traverse('postorder'):
#         if n.is_leaf(): continue
#         ch = n.get_children()
#         if len(ch) == 2:
#             l0 = ch[0].get_leaf_names()
#             l1 = ch[1].get_leaf_names()
#             s0 = {GeneToSpecies(l) for l in l0}
#             s1 = {GeneToSpecies(l) for l in l1}
#             if s0&s1:
#                 n.name = "D"
#             else:
#                 n.name = "S"
#         elif len(ch) > 2:
#             sp = [{GeneToSpecies(l) for l in c.get_leaf_names()} for c in ch]
#             if any(sp0&sp1 for sp0, sp1 in itertools.combinations(sp, 2)):
#                 n.name = "D"
#             else:
#                 n.name = "S"

# def NumberOfOrthologs(tree, GeneToSpecies):
#     species = list(set(map(GeneToSpecies, tree.get_leaf_names())))
#     genes = tree.get_leaf_names()
#     gDict = {gene:i for i,gene in enumerate(genes)}
#     sDict = {sp:i for i,sp in enumerate(species)}
#     orthologs = np.zeros((len(genes), len(species)))
#     nOrtho = 0
#     for n in tree.traverse('postorder'):
#         if n.is_leaf(): continue
#         ch = n.get_children()        
#         if len(ch) == 2: 
#             l0 = ch[0].get_leaf_names()
#             l1 = ch[1].get_leaf_names()
#             s0 = {GeneToSpecies(l) for l in l0}
#             s1 = {GeneToSpecies(l) for l in l1}
#             if s0&s1: continue
#             nOrtho += len(l0)*len(l1)
#             for g0 in l0:
#                 for s in s1:
#                     orthologs[gDict[g0], sDict[s]] = 1
#             for g1 in l1:
#                 for s in s0:
#                     orthologs[gDict[g1], sDict[s]] = 1
#         elif len(ch) > 2:
#             for ch0, ch1 in itertools.combinations(ch, 2):
#                 l0 = ch0.get_leaf_names()
#                 l1 = ch1.get_leaf_names()
#                 s0 = {GeneToSpecies(l) for l in l0}
#                 s1 = {GeneToSpecies(l) for l in l1}
#                 if s0&s1: continue
#                 nOrtho += len(l0)*len(l1)
#                 for g0 in l0:
#                     for s in s1:
#                         orthologs[gDict[g0], sDict[s]] = 1
#                 for g1 in l1:
#                     for s in s0:
#                         orthologs[gDict[g1], sDict[s]] = 1
                
#    print(nOrtho)
#    N = (len(species)-1)*len(genes)
#    print((sum(sum(orthologs)),float(N)))
#    print(sum(sum(orthologs)))

# class Finalise(object):
#     def __enter__(self):
#         pass
#     def __exit__(self, type, value, traceback):
#         ptm = parallel_task_manager.ParallelTaskManagerSingleton()
#         ptm.Stop()

# def do_trees(trees_queue, GeneToSpecies, out_dir, species_tree_rooted, qResolve):
#     # Root using species tree if provided, otherwise tree should have been rooted already
#     while True:
#         try:
#             trees_fn = trees_queue.get(True, 0.1)
#             # print("Start: " + trees_fn)
#             tree = tree_lib.Tree(trees_fn)
#         #        tree = tree_lib.Tree(trees_fn, format=3)
#             if len(tree) == 1: continue
#             if species_tree_rooted != None:
#                 tree.prune(tree.get_leaf_names())
#                 roots = om1.get_roots(tree, species_tree_rooted, GeneToSpecies)
#                 if len(roots) == 0: continue
#                 # Pick the first root for now
#                 root = roots[0]
#                 if root != tree:
#                     tree.set_outgroup(root)
#                 tree.write(outfile=(out_dir + "/" + os.path.basename(trees_fn) + "_rooted.tre"))
#             if qResolve:
#                 om1.store_species_sets(tree, GeneToSpecies)
#                 # Perform full reconciliation
#                 for n in tree.traverse("postorder"):
#                     tree = resolve(n, GeneToSpecies)
#             # NumberOfOrthologs(tree, GeneToSpecies)
#             SpeciesOverlapDuplications(tree, GeneToSpecies)
#             # print("Done: " + trees_fn)
#             tree.write(outfile=(out_dir + "/" + os.path.basename(trees_fn) + ".rec.tre"), format=3)
#         except queue.Empty:
#             return

# def Resolve_Main(trees_fn_or_dir, out_dir, species_tree_rooted_fn, GeneToSpecies, nThreads, qResolve=True, qTest=False):
#     """
#     Resolves the single tree or trees in the directory trees_fn. If no species tree is provided then the gene tree is assumed to 
#     be rooted
#     Args:
#         trees_fn - tree filename or directory containing trees
#         species_tree_rooted_fn - species tree used to root the tree. If None then the gene tree is assumed to be rooted already
#     """
#     if species_tree_rooted_fn != None:
#         species_tree_rooted = tree_lib.Tree(species_tree_rooted_fn)
#     else:
#         species_tree_rooted = None
#     qDir = True
#     try:
#         tree = tree_lib.Tree(trees_fn_or_dir)
#         qDir = False
#     except:
#         try:
#             tree = tree_lib.Tree(trees_fn_or_dir, format=3)
#             qDir = False
#         except:
#             pass
#     trees = glob.glob(trees_fn_or_dir + "/*") if qDir else [trees_fn_or_dir]

#     # for trees_fn in trees:
#     #     DoTree(trees_fn, GeneToSpecies, out_dir, species_tree_rooted, qTest)
#     queue = mp.Queue()
#     for t in trees:
#         queue.put(t)
#     # DoTree(trees_fn, GeneToSpecies, out_dir, species_tree_rooted, qTest)
#     runningProcesses = [mp.Process(target=do_trees, args=(queue, GeneToSpecies, out_dir, species_tree_rooted, qResolve)) for i_ in range(nThreads)]
#     for proc in runningProcesses:
#         proc.start()
#     for proc in runningProcesses:
#         proc.join()

# if __name__ == "__main__":
#     with Finalise():
#         from . import trees2ologs_of as om1
#         parser = argparse.ArgumentParser()
#         parser.add_argument("gene_tree")
#         parser.add_argument("out_dir")
#         parser.add_argument("-r", "--rooted_species_tree")
#         parser.add_argument("-n", "--nthreads", type=int, default=1)
#         parser.add_argument("-s", "--separator", choices=("dot", "dash", "second_dash", "3rd_dash", "hyphen"), help="Separator been species name and gene name in gene tree taxa")
#         # parser.add_argument("-t", "--test", action="store_true", help="Perform a single operation on the largest node one step down--allows testing of the method")
#         parser.add_argument("--no_resolve", action="store_true", help="Only perform species-overlap duplication analysis")
#         args = parser.parse_args()
#         Resolve_Main(args.gene_tree, args.out_dir, args.rooted_species_tree, om1.GetGeneToSpeciesMap(args), args.nthreads, not args.no_resolve, False)


# A polytomy is resolved (resolve_polytomies) only if one of its children has
# genes of several species. A polytomy of children of one species each (as
# FastTree's of identical sequences) is left as it is: its orthologs are the
# same either way, and its grouped rows are kept (tree_analyser).
RESOLVED = "resolved"
_clades = [None, None]        # (species tree, {node: its species}) of the last species tree


def resolve_polytomies(tree, GeneToSpecies, species_tree):
    """
    Resolve, in place, each polytomy of tree with a child of several species:
    its children as few groups whose children share no species as are found
    (disjoint_groups: a group's genes are then all orthologs), in a group joined as the species
    tree, and the groups joined by duplications. The children of the same
    single species are first put together, as they are in a polytomy of them
    alone. Without this, the polytomy's orthologs (every pair of children of
    no shared species) could be those of no tree. The new nodes have no
    length.

    Returns a record of each polytomy resolved, (node, genes, children,
    species sets of children), for Polytomies.tsv (tree_analyser). The node
    is the polytomy's, the top of its new nodes.
    """
    resolved = []
    for n in list(tree.traverse("postorder")):
        ch = n.get_children()
        if len(ch) <= 2:
            continue
        sets = [frozenset(s for s in map(GeneToSpecies, c.get_leaf_names()) if s is not None) for c in ch]
        if all(len(s) <= 1 for s in sets):
            continue
        resolved.append((n, n.get_leaf_names(), len(ch), len(set(sets))))
        resolve_polytomy(n, ch, sets, species_tree)
    return resolved


def resolve_polytomy(n, ch, sets, species_tree):
    """n's children ch (their species, sets) as a binary tree below n (see resolve_polytomies)."""
    units, unit_sets, by_species = [], [], {}
    for c, s in zip(ch, sets):
        c.detach()
        if len(s) == 1:
            sp = next(iter(s))
            if sp in by_species:
                by_species[sp].append(c)
                continue
            by_species[sp] = [c]
            units.append(by_species[sp])
        else:
            units.append(c)
        unit_sets.append(s)
    # (the children of a species together: a node of them, a polytomy if more than two)
    units = [(u[0] if len(u) == 1 else _new_node(u)) if isinstance(u, list) else u for u in units]
    bits = {sp: 1 << i for i, sp in enumerate(sorted(frozenset().union(*unit_sets), key=str))}
    masks = [sum(bits[sp] for sp in s) for s in unit_sets]
    groups = disjoint_groups(masks)
    clades = species_clades(species_tree)
    top = None
    for g in groups:
        sub = _species_tree_order(species_tree, [(units[i], unit_sets[i]) for i in g], clades)
        top = sub if top is None else _new_node([top, sub])
    for c in top.get_children():
        n.add_child(c.detach())


def disjoint_groups(masks):
    """
    The indices of masks (species bitmasks) as few groups of no shared species
    (DSatur colouring, the most constrained first: the fewest in nearly every
    case, not always), each sorted, by their first index. Every two groups
    share a species (a group is new only if its first one shares a species
    with every group before it).
    """
    n = len(masks)
    by_bit = defaultdict(list)
    for i, m in enumerate(masks):
        while m:
            b = m & -m
            by_bit[b].append(i)
            m ^= b
    # (the children that share a species with i, at most once each: the sum
    # over its species, an upper bound, as a tie-break)
    degree = []
    for m in masks:
        d = 0
        while m:
            b = m & -m
            d += len(by_bit[b]) - 1
            m ^= b
        degree.append(d)
    saturation = [set() for _ in range(n)]
    heap = [(0, -degree[i], i) for i in range(n)]
    heapq.heapify(heap)
    done = [False] * n
    unions, members = [], []
    while heap:
        neg_sat, _, i = heapq.heappop(heap)
        if done[i] or -neg_sat != len(saturation[i]):
            continue            # (an entry since replaced)
        done[i] = True
        g = next((k for k, u in enumerate(unions) if not u & masks[i]), None)
        if g is None:
            unions.append(0)
            members.append([])
            g = len(unions) - 1
        unions[g] |= masks[i]
        members[g].append(i)
        m = masks[i]
        while m:
            b = m & -m
            m ^= b
            for j in by_bit[b]:
                if not done[j] and g not in saturation[j]:
                    saturation[j].add(g)
                    heapq.heappush(heap, (-len(saturation[j]), -degree[j], j))
    return sorted(sorted(m) for m in members)


def species_clades(species_tree):
    """{node: frozenset of its species} of the species tree (kept for the next tree)."""
    if _clades[0] is not species_tree:
        clades = {}
        for node in species_tree.traverse("postorder"):
            clades[node] = (frozenset([node.name]) if node.is_leaf() else
                            frozenset().union(*(clades[c] for c in node.get_children())))
        _clades[:] = [species_tree, clades]
    return _clades[1]


def _species_tree_order(st_node, items, clades):
    """items ((node, species), of no shared species) joined as the species tree below st_node."""
    if len(items) == 1:
        return items[0][0]
    parts = []
    spanning = list(items)
    for c in st_node.get_children():
        below = [it for it in spanning if it[1] and it[1] <= clades[c]]
        if below:
            parts.append(_species_tree_order(c, below, clades))
            spanning = [it for it in spanning if not (it[1] and it[1] <= clades[c])]
    parts.extend(node for node, _ in spanning)
    top = parts[0]
    for p in parts[1:]:
        top = _new_node([top, p])
    return top


def _new_node(children):
    node = tree_lib.TreeNode()
    node.dist = 0.
    for c in children:
        node.add_child(c)
    return node


def resolve_tree(tree, GeneToSpecies):
    if tree is None:
        raise ValueError("Resolve got tree=None")
    # (only the species below each node are used: not those above, nearly all
    # of them at every node)
    store_species_sets(tree, GeneToSpecies, q_up=False)
    for n in tree.traverse("postorder"):
        new_tree = resolve(n, GeneToSpecies)
        if new_tree is not None:
            tree = new_tree
    return tree
