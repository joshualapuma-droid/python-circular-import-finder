#!/usr/bin/env python3
"""
CIRCULAR IMPORT DETECTION & HEALING
====================================

Because everyone hits this at some point and wants to scream.

Features:
- Hooks into import machinery
- Detects circular imports before they crash
- Suggests refactors or lazy imports
- Provides import_maximum() that auto-lazifies when cycles detected
- Visualizes import dependency graph
"""

import importlib
import importlib.util
import sys
import threading
import warnings
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple


class CircularImportDetected(ImportWarning):
    """Warning for circular import detection"""
    pass


class ImportTracker:
    """Tracks import dependencies to detect cycles"""
    
    def __init__(self):
        self.import_graph = defaultdict(set)  # module -> set of modules it imports
        self.import_stack = []  # Current import chain
        self.cycles_detected = []
        self.lock = threading.RLock()
    
    def start_import(self, module_name: str):
        """Mark start of importing a module"""
        with self.lock:
            # Check if this creates a cycle
            if module_name in self.import_stack:
                cycle = self.import_stack[self.import_stack.index(module_name):] + [module_name]
                self.cycles_detected.append(cycle)
                return cycle
            
            self.import_stack.append(module_name)
            return None
    
    def finish_import(self, module_name: str):
        """Mark end of importing a module"""
        with self.lock:
            if self.import_stack and self.import_stack[-1] == module_name:
                self.import_stack.pop()
    
    def add_dependency(self, importer: str, imported: str):
        """Record that 'importer' imports 'imported'"""
        with self.lock:
            self.import_graph[importer].add(imported)
    
    def find_cycles(self) -> List[List[str]]:
        """Find all cycles in the import graph using DFS"""
        def dfs(node: str, visited: Set[str], rec_stack: List[str]) -> List[List[str]]:
            visited.add(node)
            rec_stack.append(node)
            cycles = []
            
            for neighbor in self.import_graph.get(node, []):
                if neighbor not in visited:
                    cycles.extend(dfs(neighbor, visited, rec_stack[:]))
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = rec_stack.index(neighbor)
                    cycle = rec_stack[cycle_start:] + [neighbor]
                    cycles.append(cycle)
            
            return cycles
        
        all_cycles = []
        visited = set()
        
        for node in self.import_graph:
            if node not in visited:
                cycles = dfs(node, visited, [])
                all_cycles.extend(cycles)
        
        # Deduplicate cycles
        unique_cycles = []
        seen = set()
        for cycle in all_cycles:
            # Normalize cycle (rotate to start with smallest element)
            min_idx = cycle.index(min(cycle[:-1]))  # Exclude last (duplicate) element
            normalized = tuple(cycle[min_idx:-1] + cycle[:min_idx] + [cycle[min_idx]])
            if normalized not in seen:
                seen.add(normalized)
                unique_cycles.append(list(normalized))
        
        return unique_cycles
    
    def get_import_depth(self, module_name: str) -> int:
        """Get the depth of imports for a module"""
        def bfs_depth(start: str) -> int:
            if start not in self.import_graph:
                return 0
            
            queue = deque([(start, 0)])
            visited = {start}
            max_depth = 0
            
            while queue:
                node, depth = queue.popleft()
                max_depth = max(max_depth, depth)
                
                for neighbor in self.import_graph.get(node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))
            
            return max_depth
        
        return bfs_depth(module_name)


# Global import tracker
_import_tracker = ImportTracker()


class CircularImportFinder:
    """Custom import hook to detect circular imports"""
    
    def find_spec(self, name, path, target=None):
        """Called when trying to import a module"""
        # Check for circular import
        cycle = _import_tracker.start_import(name)
        
        if cycle:
            # Circular import detected!
            cycle_str = " → ".join(cycle)
            warnings.warn(
                f"🔄 Circular import detected: {cycle_str}",
                CircularImportDetected,
                stacklevel=2
            )
            
            # Suggest lazy import
            print(f"\n💡 Suggestion: Use lazy import to break the cycle:")
            print(f"   Instead of: import {cycle[-2]}")
            print(f"   Use: from . import {cycle[-2]} (inside function)")
            print(f"   Or: importlib.import_module('{cycle[-2]}')")
        
        # Let default import machinery handle it
        return None
    
    def find_module(self, fullname, path=None):
        """Legacy hook for older Python versions"""
        self.find_spec(fullname, path)
        return None


def install_circular_import_detector():
    """Install the circular import detector"""
    if not any(isinstance(finder, CircularImportFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, CircularImportFinder())
        print("✅ Circular import detector installed")


def import_maximum(module_name: str, lazy: bool = False) -> Any:
    """
    Smart import that handles circular dependencies.
    
    Args:
        module_name: Name of module to import
        lazy: If True, always use lazy import
    
    Returns:
        Imported module or lazy loader
    """
    # Track this import
    caller_frame = sys._getframe(1)
    caller_module = caller_frame.f_globals.get('__name__', '__main__')
    
    # Check if this would create a cycle
    cycle = _import_tracker.start_import(module_name)
    
    if cycle or lazy:
        # Use lazy import
        print(f"🔄 Using lazy import for {module_name}")
        return LazyModule(module_name)
    
    # Normal import
    try:
        module = importlib.import_module(module_name)
        _import_tracker.add_dependency(caller_module, module_name)
        _import_tracker.finish_import(module_name)
        return module
    except Exception as e:
        _import_tracker.finish_import(module_name)
        raise


class LazyModule:
    """Lazy-loading module wrapper"""
    
    def __init__(self, module_name: str):
        self._module_name = module_name
        self._module = None
        self._loading = False
    
    def _load(self):
        """Actually load the module"""
        if self._module is None and not self._loading:
            self._loading = True
            try:
                self._module = importlib.import_module(self._module_name)
                _import_tracker.finish_import(self._module_name)
            finally:
                self._loading = False
    
    def __getattr__(self, name):
        """Load module on first attribute access"""
        self._load()
        return getattr(self._module, name)
    
    def __dir__(self):
        """Support dir() introspection"""
        self._load()
        return dir(self._module)
    
    def __repr__(self):
        if self._module is None:
            return f"<LazyModule '{self._module_name}' (not loaded)>"
        return f"<LazyModule '{self._module_name}' (loaded)>"


def analyze_imports(module_name: str = None) -> Dict[str, Any]:
    """Analyze import dependencies"""
    if module_name:
        # Analyze specific module
        depth = _import_tracker.get_import_depth(module_name)
        dependencies = list(_import_tracker.import_graph.get(module_name, []))
        
        return {
            'module': module_name,
            'import_depth': depth,
            'direct_dependencies': dependencies,
            'dependency_count': len(dependencies)
        }
    
    # Analyze all imports
    cycles = _import_tracker.find_cycles()
    
    return {
        'total_modules': len(_import_tracker.import_graph),
        'total_dependencies': sum(len(deps) for deps in _import_tracker.import_graph.values()),
        'circular_dependencies': len(cycles),
        'cycles': cycles,
        'most_imported': sorted(
            [(mod, len(deps)) for mod, deps in _import_tracker.import_graph.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
    }


def visualize_import_graph(output_file: str = "import_graph.dot"):
    """Generate GraphViz visualization of import graph"""
    try:
        lines = ["digraph ImportGraph {"]
        lines.append('  rankdir=LR;')
        lines.append('  node [shape=box];')
        
        # Find cycles first
        cycles = _import_tracker.find_cycles()
        cycle_edges = set()
        for cycle in cycles:
            for i in range(len(cycle) - 1):
                cycle_edges.add((cycle[i], cycle[i+1]))
        
        # Add edges
        for module, dependencies in _import_tracker.import_graph.items():
            for dep in dependencies:
                # Highlight cycle edges in red
                if (module, dep) in cycle_edges:
                    lines.append(f'  "{module}" -> "{dep}" [color=red, penwidth=2];')
                else:
                    lines.append(f'  "{module}" -> "{dep}";')
        
        lines.append("}")
        
        with open(output_file, 'w') as f:
            f.write("\n".join(lines))
        
        print(f"✅ Import graph saved to {output_file}")
        print(f"   Generate PNG with: dot -Tpng {output_file} -o import_graph.png")
        
    except Exception as e:
        print(f"❌ Failed to create visualization: {e}")


def suggest_refactors() -> List[Dict[str, Any]]:
    """Suggest refactorings to reduce circular dependencies"""
    suggestions = []
    
    cycles = _import_tracker.find_cycles()
    
    for cycle in cycles:
        suggestion = {
            'cycle': cycle,
            'options': []
        }
        
        # Option 1: Move shared code to a new module
        suggestion['options'].append({
            'type': 'extract_module',
            'description': f"Extract shared code into a new module imported by all",
            'example': f"Create 'common.py' and move shared code there"
        })
        
        # Option 2: Use lazy imports
        suggestion['options'].append({
            'type': 'lazy_import',
            'description': f"Import inside functions instead of at module level",
            'example': f"In {cycle[-2]}, move 'import {cycle[-1]}' inside function"
        })
        
        # Option 3: Dependency injection
        suggestion['options'].append({
            'type': 'dependency_injection',
            'description': f"Pass dependencies as function parameters",
            'example': f"Instead of importing, pass objects as parameters"
        })
        
        suggestions.append(suggestion)
    
    return suggestions


def circular_import_report():
    """Generate comprehensive circular import report"""
    print("\n" + "=" * 70)
    print("🔄 CIRCULAR IMPORT ANALYSIS")
    print("=" * 70)
    
    analysis = analyze_imports()
    
    print(f"\n📊 Statistics:")
    print(f"  Total Modules Tracked: {analysis['total_modules']}")
    print(f"  Total Dependencies: {analysis['total_dependencies']}")
    print(f"  Circular Dependencies: {analysis['circular_dependencies']}")
    
    if analysis['cycles']:
        print(f"\n⚠️  Detected Cycles:")
        for i, cycle in enumerate(analysis['cycles'], 1):
            cycle_str = " → ".join(cycle)
            print(f"  {i}. {cycle_str}")
    
    if analysis['most_imported']:
        print(f"\n📈 Most Complex Modules (by dependencies):")
        for module, count in analysis['most_imported'][:5]:
            print(f"  • {module}: {count} imports")
    
    # Suggestions
    if analysis['cycles']:
        print(f"\n💡 Suggested Refactors:")
        suggestions = suggest_refactors()
        for i, suggestion in enumerate(suggestions[:3], 1):
            print(f"\n  Cycle {i}: {' → '.join(suggestion['cycle'])}")
            for opt in suggestion['options'][:2]:
                print(f"    ✓ {opt['description']}")
                print(f"      Example: {opt['example']}")


def demo_circular_import_detection():
    """Demonstrate circular import detection"""
    print("🔄 Circular Import Detection Demo\n")
    
    # Install detector
    install_circular_import_detector()
    
    # Simulate some imports
    print("Simulating import dependencies...\n")
    
    _import_tracker.add_dependency('module_a', 'module_b')
    _import_tracker.add_dependency('module_b', 'module_c')
    _import_tracker.add_dependency('module_c', 'module_a')  # Creates cycle!
    
    _import_tracker.add_dependency('module_d', 'module_e')
    _import_tracker.add_dependency('module_e', 'module_f')
    
    # Generate report
    circular_import_report()
    
    # Demonstrate import_maximum
    print("\n📦 Using import_maximum:")
    print("  lazy_mod = import_maximum('some_module', lazy=True)")
    print("  → Returns LazyModule that loads on first use")


if __name__ == "__main__":
    demo_circular_import_detection()
