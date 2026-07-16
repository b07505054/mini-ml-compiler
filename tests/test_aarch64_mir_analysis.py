#!/usr/bin/env python3
"""test_aarch64_mir_analysis.py

Host-side unit tests (Stage 18) for tools/analyze_aarch64_candidate_mir.py
and tools/extract_aarch64_candidate_mir.py, using small synthetic MIR
fixtures (not full committed MIR dumps) so they run instantly with no
network access and no Raspberry Pi.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

import analyze_aarch64_candidate_mir as mir  # noqa: E402


# A small synthetic pre-RA MIR fixture: one machine function, one ciface
# wrapper, virtual registers with explicit classes, no stack objects yet.
PRE_RA_FIXTURE = """--- |
  define void @kernel_tm4_tn4_tk4() {
    ret void
  }
...
---
name:            kernel_tm4_tn4_tk4
registers:
  - { id: 0, class: gpr64, preferred-register: '', flags: [  ] }
  - { id: 1, class: fpr128, preferred-register: '', flags: [  ] }
  - { id: 2, class: fpr128, preferred-register: '', flags: [  ] }
  - { id: 3, class: fpr128, preferred-register: '', flags: [  ] }
liveins:         []
frameInfo:
  stackSize:       0
  maxAlignment:    16
  hasCalls:        true
  adjustsStack:    true
stack:           []
fixedStack:      []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo: {}
body:             |
  bb.0:
    %0:gpr64 = COPY $x0
    %1:fpr128 = LDRQui %0, 0
    %2:fpr128 = LDRQui %0, 1
    %3:fpr128 = nofpexcept FMLAv4i32_indexed killed %1, %2, %1, 0
    RET_ReallyLR
...
---
name:            _mlir_ciface_kernel_tm4_tn4_tk4
registers:
  - { id: 0, class: gpr64, preferred-register: '', flags: [  ] }
liveins:         []
frameInfo:
  stackSize:       0
stack:           []
fixedStack:      []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo: {}
body:             |
  bb.0:
    %0:gpr64 = COPY $x0
    RET_ReallyLR
...
"""

# A synthetic post-RA MIR fixture with ONE real allocator spill (empty
# callee-saved-register) and 12 callee-saved (ABI) slots that also carry
# `type: spill-slot` -- exercising the exact classification distinction
# this tool exists to get right.
POST_RA_FIXTURE = """--- |
  define void @kernel_tm8_tn8_tk8() {
    ret void
  }
...
---
name:            kernel_tm8_tn8_tk8
registers:       []
liveins:         []
frameInfo:
  stackSize:       112
  maxAlignment:    16
  hasCalls:        true
  adjustsStack:    true
stack:
  - { id: 0, name: '', type: spill-slot, offset: -8, size: 8, alignment: 8,
      stack-id: default, callee-saved-register: '$x19', callee-saved-restored: true,
      debug-info-variable: '', debug-info-expression: '', debug-info-location: '' }
  - { id: 1, name: '', type: spill-slot, offset: -16, size: 16, alignment: 16,
      stack-id: default, callee-saved-register: '', callee-saved-restored: true,
      debug-info-variable: '', debug-info-expression: '', debug-info-location: '' }
fixedStack:
  - { id: 0, type: default, offset: 96, size: 8, alignment: 16, stack-id: default,
      isImmutable: true, isAliased: false, callee-saved-register: '', callee-saved-restored: true,
      debug-info-variable: '', debug-info-expression: '', debug-info-location: '' }
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo: {}
body:             |
  bb.0:
    liveins: $x19
    $q0 = LDRQui $x0, 0
    $q1 = LDRQui $x0, 1
    STRQui killed $q0, %stack.1, 0 :: (store (s128) into %stack.1)
    $q2 = nofpexcept FMLAv4i32_indexed killed $q1, $q1, $q1, 0
    $q0 = LDRQui %stack.1, 0 :: (load (s128) from %stack.1)
    RET_ReallyLR
...
"""


def write_fixture(text):
    fd, path = tempfile.mkstemp(suffix=".mir")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return path


class MachineFunctionSplitting(unittest.TestCase):
    def test_finds_both_functions(self):
        functions = mir.split_machine_functions(PRE_RA_FIXTURE)
        names = [n for n, _ in functions]
        self.assertIn("kernel_tm4_tn4_tk4", names)
        self.assertIn("_mlir_ciface_kernel_tm4_tn4_tk4", names)

    def test_picks_kernel_not_wrapper(self):
        functions = mir.split_machine_functions(PRE_RA_FIXTURE)
        name, _ = mir.pick_kernel_function(functions)
        self.assertEqual(name, "kernel_tm4_tn4_tk4")


class VirtualRegisterParsing(unittest.TestCase):
    def test_parses_all_registers_with_classes(self):
        _, doc = mir.pick_kernel_function(mir.split_machine_functions(PRE_RA_FIXTURE))
        regs = mir.parse_registers(doc)
        self.assertEqual(len(regs), 4)
        self.assertEqual(regs[0], "gpr64")
        self.assertEqual(regs[1], "fpr128")


class RegisterClassParsing(unittest.TestCase):
    def test_vector_vs_gpr_classification(self):
        path = write_fixture(PRE_RA_FIXTURE)
        try:
            result = mir.analyze_stage(path, "pre_ra")
        finally:
            os.unlink(path)
        self.assertEqual(result["virtual_vector_registers"], 3)  # 3x fpr128
        self.assertEqual(result["virtual_gpr_registers"], 1)  # 1x gpr64
        self.assertEqual(result["register_class_distribution"], {"gpr64": 1, "fpr128": 3})


class FrameObjectParsing(unittest.TestCase):
    """Item 6: frame-object parsing, including the multi-line YAML entries
    LLVM emits (this is the exact bug class found and fixed during this
    slice's development -- see the module's multi-line join logic)."""

    def test_parses_multiline_stack_entries(self):
        _, doc = mir.pick_kernel_function(mir.split_machine_functions(POST_RA_FIXTURE))
        objs = mir.parse_stack_objects(doc, "stack")
        self.assertEqual(len(objs), 2)
        self.assertEqual(objs[0]["callee-saved-register"], "$x19")
        self.assertEqual(objs[1]["callee-saved-register"], "")

    def test_parses_fixed_stack(self):
        _, doc = mir.pick_kernel_function(mir.split_machine_functions(POST_RA_FIXTURE))
        objs = mir.parse_stack_objects(doc, "fixedStack")
        self.assertEqual(len(objs), 1)


class SpillReloadClassification(unittest.TestCase):
    """Item 7: spill/reload classification; item 8: ABI save/restore
    exclusion -- the central correctness property of this tool."""

    def test_real_spill_counted(self):
        path = write_fixture(POST_RA_FIXTURE)
        try:
            result = mir.analyze_stage(path, "post_ra")
        finally:
            os.unlink(path)
        self.assertEqual(result["spill_slot_count"], 1)
        self.assertEqual(result["spill_slot_bytes"], 16)
        self.assertEqual(result["spill_stores"], 1)
        self.assertEqual(result["spill_reloads"], 1)

    def test_callee_saved_slot_excluded_from_spill_count(self):
        """The fixture has 2 `type: spill-slot` stack entries, but only 1
        has an empty callee-saved-register -- the other MUST NOT be
        counted as an allocator spill despite sharing the same `type`."""
        path = write_fixture(POST_RA_FIXTURE)
        try:
            result = mir.analyze_stage(path, "post_ra")
        finally:
            os.unlink(path)
        self.assertEqual(result["callee_saved_stack_slots"], 1)
        self.assertEqual(result["frame_objects_total"], 2)
        self.assertLess(result["spill_slot_count"], result["frame_objects_total"])

    def test_reload_detected_despite_destination_operand(self):
        """Regression test for the specific bug found in development: a
        reload's opcode is NOT line-initial (`$q0 = LDRQui ...`), unlike a
        spill store (`STRQui $q0, ...`), which IS line-initial."""
        path = write_fixture(POST_RA_FIXTURE)
        try:
            result = mir.analyze_stage(path, "post_ra")
        finally:
            os.unlink(path)
        self.assertGreaterEqual(result["spill_reloads"], 1)


class CopyCounting(unittest.TestCase):
    def test_counts_copy_opcode(self):
        path = write_fixture(PRE_RA_FIXTURE)
        try:
            result = mir.analyze_stage(path, "pre_ra")
        finally:
            os.unlink(path)
        self.assertEqual(result["copies"], 1)


class NullHandlingForUnavailableMetrics(unittest.TestCase):
    def test_approx_peak_present_for_pre_ra(self):
        path = write_fixture(PRE_RA_FIXTURE)
        try:
            result = mir.analyze_stage(path, "pre_ra")
        finally:
            os.unlink(path)
        self.assertIsNotNone(result["approx_peak_live_vector_registers"])

    def test_approx_peak_none_when_no_registers_yaml(self):
        """Post-RA MIR has no `registers:` list (all vregs eliminated) --
        the approximation must degrade to None, never a fabricated 0."""
        path = write_fixture(POST_RA_FIXTURE)
        try:
            result = mir.analyze_stage(path, "post_ra")
        finally:
            os.unlink(path)
        self.assertIsNone(result["approx_peak_live_vector_registers"])


class JsonSchemaCompatibility(unittest.TestCase):
    def test_output_is_json_serializable(self):
        path = write_fixture(PRE_RA_FIXTURE)
        try:
            result = mir.analyze_stage(path, "pre_ra")
        finally:
            os.unlink(path)
        json.dumps(result)  # must not raise

    def test_comparison_keys_present_when_both_stages_given(self):
        pre_path = write_fixture(PRE_RA_FIXTURE)
        post_path = write_fixture(POST_RA_FIXTURE)
        try:
            pre = mir.analyze_stage(pre_path, "pre_ra")
            post = mir.analyze_stage(post_path, "post_ra")
        finally:
            os.unlink(pre_path)
            os.unlink(post_path)
        for key in ("virtual_registers_total", "virtual_vector_registers"):
            self.assertIn(key, pre)
        for key in ("spill_stores", "spill_reloads", "spill_slot_count"):
            self.assertIn(key, post)


class ExtractionCommandGeneration(unittest.TestCase):
    """Item 1: MIR extraction command generation -- verifies the extractor
    script's --stop-after pass names match Stage-1 discovery, without
    actually invoking llc (fast, no toolchain dependency)."""

    def test_stage_passes_match_discovered_names(self):
        sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
        import extract_aarch64_candidate_mir as extractor
        self.assertEqual(extractor.STAGE_PASSES["post_isel"], "finalize-isel")
        self.assertEqual(extractor.STAGE_PASSES["pre_ra"], "machine-scheduler")
        self.assertEqual(extractor.STAGE_PASSES["post_ra"], "virtregrewriter")
        self.assertEqual(extractor.STAGE_PASSES["post_prologue_epilogue"], "prologepilog")

    def test_cli_requires_all_mandatory_args(self):
        proc = subprocess.run(
            ["python3", os.path.join(REPO_ROOT, "tools", "extract_aarch64_candidate_mir.py")],
            capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)


if __name__ == "__main__":
    unittest.main()
