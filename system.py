import sys
import csv
import itertools
from collections import defaultdict

def load_dataset(filename):
    """Reads the input CSV and returns baskets as list of sets."""
    baskets = []
    with open(filename, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            clean_items = [item.strip() for item in row if item.strip()]
            baskets.append(set(clean_items))
    return baskets

def get_frequent_itemsets(baskets, min_sup):
    """Generate frequent itemsets using Apriori algorithm with join and prune steps."""
    total_baskets = len(baskets)
    support_data = dict()

    # Step 1: Find frequent 1-itemsets 
    count = defaultdict(int)
    for basket in baskets:
        for item in basket:
            count[frozenset([item])] += 1
    L1 = {itemset: cnt / total_baskets for itemset, cnt in count.items() if cnt / total_baskets >= min_sup}
    support_data.update(L1)
    current_L = L1
    k = 2

    while current_L:
        prev_itemsets = list(current_L.keys())
        candidates = set()

        # --- JOIN STEP  ---
        for i in range(len(prev_itemsets)):
            for j in range(i + 1, len(prev_itemsets)):
                l1, l2 = list(prev_itemsets[i]), list(prev_itemsets[j])
                l1.sort()
                l2.sort()
                if l1[:k-2] == l2[:k-2]:  # First k-2 items match
                    candidate = prev_itemsets[i] | prev_itemsets[j]
                    if len(candidate) == k:
                        candidates.add(candidate)

        # --- PRUNE STEP ---
        pruned_candidates = set()
        for candidate in candidates:
            # Generate all (k-1)-subsets of candidate
            subsets = itertools.combinations(candidate, k-1)
            all_subsets_exist = True
            for subset in subsets:
                if frozenset(subset) not in current_L:
                    all_subsets_exist = False
                    break
            if all_subsets_exist:
                pruned_candidates.add(candidate)

        # --- COUNT SUPPORT FOR PRUNED CANDIDATES ---
        candidate_count = defaultdict(int)
        for basket in baskets:
            for candidate in pruned_candidates:
                if candidate.issubset(basket):
                    candidate_count[candidate] += 1

        current_L = {
            itemset: cnt / total_baskets 
            for itemset, cnt in candidate_count.items() 
            if cnt / total_baskets >= min_sup
        }
        support_data.update(current_L)
        k += 1

    return support_data

def generate_rules(support_data, min_conf):
    """Generate association rules from frequent itemsets meeting minimum confidence."""
    rules = []
    for itemset in support_data:
        if len(itemset) >= 2:
            for rhs in itemset:
                lhs = itemset - frozenset([rhs])
                if lhs in support_data:
                    conf = support_data[itemset] / support_data[lhs]
                    if conf >= min_conf:
                        rules.append((lhs, frozenset([rhs]), support_data[itemset], conf))
    return rules

def write_output(support_data, rules, min_sup, min_conf):
    """Write frequent itemsets and association rules to output.txt in specified format."""
    with open("output.txt", "w", encoding='utf-8') as f:
        # Write frequent itemsets header
        f.write(f"==Frequent itemsets (min_sup={min_sup*100:.0f}%)\n")
        
        # Sort itemsets by support in descending order and then by itemset length
        sorted_itemsets = sorted(support_data.items(), 
                               key=lambda x: (-x[1], len(x[0]), x[0]))
        
        # Write itemsets with percentage sign and no decimal places
        for itemset, support in sorted_itemsets:
            items = sorted(list(itemset))
            f.write(f"[{','.join(items)}], {support*100:.0f}%\n")
        
        # Write rules header
        f.write(f"\n==High-confidence association rules (min_conf={min_conf*100:.0f}%)\n")
        
        # Sort rules by confidence in descending order
        sorted_rules = sorted(rules, key=lambda x: -x[3])
        
        # Write rules with specified decimal places
        for lhs, rhs, support, confidence in sorted_rules:
            lhs_str = ",".join(sorted(lhs))
            rhs_str = ",".join(rhs)  # rhs is a single item
            f.write(f"[{lhs_str}] => [{rhs_str}] (Conf: {confidence*100:.1f}%, Supp: {support*100:.1f}%)\n")

def main():
    """Execute the Apriori algorithm: load data, find itemsets, generate rules, and output results."""
    if len(sys.argv) != 4:
        print("Usage: python3 main.py INTEGRATED-DATASET.csv min_sup min_conf")
        sys.exit(1)

    dataset_file = sys.argv[1]
    min_sup = float(sys.argv[2])
    min_conf = float(sys.argv[3])

    print("Loading dataset...")
    baskets = load_dataset(dataset_file)

    print("Running Apriori...")
    support_data = get_frequent_itemsets(baskets, min_sup)

    print("Generating rules...")
    rules = generate_rules(support_data, min_conf)

    print("Writing output to output.txt...")
    write_output(support_data, rules, min_sup, min_conf)
    print("Done!")

if __name__ == "__main__":
    main()