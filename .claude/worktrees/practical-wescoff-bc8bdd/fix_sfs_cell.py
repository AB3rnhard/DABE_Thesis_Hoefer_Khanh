import json

path = "Feature Engineering SQL polymarket database.ipynb"
with open(path, encoding="utf-8") as f:
    nb = json.load(f)

# Find cell with X_SFS_mi
target_cell = None
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "X_SFS_mi = df_full_non0var_relevant" in src and "sfs = SFS(rf" in src:
        target_cell = i
        break

if target_cell is None:
    print("ERROR: SFS cell not found")
    exit(1)

print(f"Found SFS cell at index {target_cell} (cell #{target_cell+1})")

cell = nb["cells"][target_cell]
old_src = list(cell["source"])

# Print current source for inspection
print("--- CURRENT SOURCE ---")
for line in old_src:
    print(repr(line))

new_source = []
i = 0
while i < len(old_src):
    line = old_src[i]
    
    # Fix 1: replace .loc with .reindex for y alignment
    if 'y = gold["y"].loc[X_SFS_mi.index]' in line:
        new_source.append("# Use reindex (not .loc) so timestamps only in Polymarket but not Bloomberg get NaN instead of KeyError\n")
        new_source.append('y = gold["y"].reindex(X_SFS_mi.index)\n')
        i += 1
        continue
    
    # Fix 2: update the NaN drop comment
    if "# Drop rows where y is NaN (the last HORIZON_STEPS rows are always NaN due to the forward shift)" in line:
        new_source.append("# Drop rows where y is NaN (last HORIZON_STEPS rows are NaN due to forward shift,\n")
        new_source.append("# plus any Polymarket timestamps outside the Bloomberg gold coverage window)\n")
        i += 1
        continue
    
    # Fix 3: add NaN imputation after the column-flattening line
    if 'X_SFS.columns = [f"{var}__{mkt}" for var, mkt in X_SFS_mi.columns]' in line:
        new_source.append(line)
        i += 1
        # Insert imputation block after this line (skip the existing blank line if any)
        if i < len(old_src) and old_src[i].strip() == "":
            i += 1  # skip blank line
        new_source.append("\n")
        new_source.append("# RandomForest cannot handle NaN values; forward-fill (carry last known value forward in\n")
        new_source.append("# time) then fill any still-missing entries with the column median (for columns NaN from start)\n")
        new_source.append("from sklearn.impute import SimpleImputer\n")
        new_source.append("X_SFS = X_SFS.ffill().bfill()\n")
        new_source.append("imputer = SimpleImputer(strategy='median')\n")
        new_source.append("X_SFS = pd.DataFrame(imputer.fit_transform(X_SFS), index=X_SFS.index, columns=X_SFS.columns)\n")
        new_source.append("\n")
        continue
    
    new_source.append(line)
    i += 1

nb["cells"][target_cell]["source"] = new_source

print("\n--- NEW SOURCE ---")
for line in new_source:
    print(repr(line))

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("\nDone!")
