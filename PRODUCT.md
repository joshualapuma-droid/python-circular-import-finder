# Circular Import Finder - Detect Import Cycles

**Find and fix circular imports before they cause runtime errors.**

## Product Overview

Circular Import Finder analyzes your Python project to detect circular import dependencies, visualizes import graphs, and suggests refactoring solutions.

## Key Features

✅ **Cycle Detection** - Find all circular imports  
✅ **Visual Graphs** - See import relationships  
✅ **Refactoring Hints** - Suggestions to fix cycles  
✅ **Static Analysis** - No code execution needed  
✅ **Fast Scanning** - Analyze large projects quickly  
✅ **CI/CD Integration** - Prevent new cycles  

## Quick Start

```python
from enhanced_circular_import_detector import CircularImportFinder

finder = CircularImportFinder()

# Scan project
cycles = finder.find_cycles("/project/src")

# Display results
for cycle in cycles:
    print(f"Circular import: {' -> '.join(cycle)}")

# Visualize
finder.visualize_graph("import_graph.png")

# Get refactoring suggestions
suggestions = finder.suggest_fixes(cycles[0])
# "Move shared code to utils.py"
# "Use TYPE_CHECKING import"
# "Delay import until function call"
```

## Shopify Product Details

**Product Title:** Circular Import Finder - Detect Python Import Cycles

**Price:** $79

**Tags:** imports, circular-dependencies, static-analysis, refactoring, architecture

**Short Description:**
Detect circular imports in Python projects. Visualize import graphs, get refactoring suggestions. Prevent runtime import errors. CI/CD ready.

**Buy Now - $79**
