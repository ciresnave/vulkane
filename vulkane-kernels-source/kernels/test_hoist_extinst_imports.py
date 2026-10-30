#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Characterisation tests for hoist_extinst_imports.hoist_imports.

These pin what the function does TODAY, measured by running it on each fixture
(2026-10-30), so that splitting `hoist_imports` (ROADMAP row 1d, Codacy CC 34)
can be shown to change nothing. They are text-in/text-out: no spirv-dis,
spirv-as or spirv-val is needed.

Run:  python3 -m unittest test_hoist_extinst_imports   (from this directory)
"""

import unittest

import hoist_extinst_imports as h

HEADER = """\
               OpCapability Shader
               OpCapability Float64
         %gl = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
"""

BODY_OPEN = """\
         %main = OpFunction %void None %fn
         %entry = OpLabel
"""

BODY_CLOSE = """\
               OpReturn
               OpFunctionEnd
"""


class HoistImports(unittest.TestCase):
    def test_no_in_function_import_is_unchanged(self):
        text = HEADER + BODY_OPEN + "         %r = OpExtInst %float %gl 27 %x\n" + BODY_CLOSE
        self.assertEqual(h.hoist_imports(text), text)

    def test_in_function_import_moves_to_module_scope_after_last_import(self):
        text = (
            HEADER
            + BODY_OPEN
            + '         %cl = OpExtInstImport "OpenCL.std"\n'
            + "         %r = OpExtInst %double %cl 19 %x\n"
            + BODY_CLOSE
        )
        expected = (
            """\
               OpCapability Shader
               OpCapability Float64
         %gl = OpExtInstImport "GLSL.std.450"
         %cl = OpExtInstImport "OpenCL.std"
               OpMemoryModel Logical GLSL450
"""
            + BODY_OPEN
            + "         %r = OpExtInst %double %cl 19 %x\n"
            + BODY_CLOSE
        )
        self.assertEqual(h.hoist_imports(text), expected)

    def test_duplicates_dedupe_and_references_follow_the_first_seen_id(self):
        # Both %cl_1 and %cl_10 are aliases of the module-scope %cl, and they
        # share a prefix: rewriting %cl_1 without the word boundary (or before
        # %cl_10) would turn %cl_10 into %cl0.
        text = (
            "               OpCapability Shader\n"
            '         %cl = OpExtInstImport "OpenCL.std"\n'
            "               OpMemoryModel Logical GLSL450\n"
            "         %f = OpFunction %void None %fn\n"
            "         %e = OpLabel\n"
            '         %cl_1 = OpExtInstImport "OpenCL.std"\n'
            '         %cl_10 = OpExtInstImport "OpenCL.std"\n'
            "         %a = OpExtInst %double %cl_1 19 %x\n"
            "         %b = OpExtInst %double %cl_10 19 %x\n"
            "               OpFunctionEnd\n"
        )
        expected = (
            "               OpCapability Shader\n"
            '         %cl = OpExtInstImport "OpenCL.std"\n'
            "               OpMemoryModel Logical GLSL450\n"
            "         %f = OpFunction %void None %fn\n"
            "         %e = OpLabel\n"
            "         %a = OpExtInst %double %cl 19 %x\n"
            "         %b = OpExtInst %double %cl 19 %x\n"
            "               OpFunctionEnd\n"
        )
        self.assertEqual(h.hoist_imports(text), expected)

    def test_module_scope_import_wins_over_a_later_in_function_duplicate(self):
        text = (
            HEADER
            + BODY_OPEN
            + '         %gl_again = OpExtInstImport "GLSL.std.450"\n'
            + "         %r = OpExtInst %float %gl_again 27 %x\n"
            + BODY_CLOSE
        )
        expected = HEADER + BODY_OPEN + "         %r = OpExtInst %float %gl 27 %x\n" + BODY_CLOSE
        self.assertEqual(h.hoist_imports(text), expected)

    def test_with_no_module_imports_the_new_one_goes_after_the_last_capability(self):
        text = (
            "               OpCapability Shader\n"
            "               OpCapability Float64\n"
            "               OpMemoryModel Logical GLSL450\n"
            + BODY_OPEN
            + '         %cl = OpExtInstImport "OpenCL.std"\n'
            + BODY_CLOSE
        )
        out = h.hoist_imports(text).splitlines()
        self.assertEqual(
            out[:4],
            [
                "               OpCapability Shader",
                "               OpCapability Float64",
                '         %cl = OpExtInstImport "OpenCL.std"',
                "               OpMemoryModel Logical GLSL450",
            ],
        )

    def test_idempotent(self):
        text = (
            HEADER
            + BODY_OPEN
            + '         %cl = OpExtInstImport "OpenCL.std"\n'
            + BODY_CLOSE
        )
        once = h.hoist_imports(text)
        self.assertEqual(h.hoist_imports(once), once)

    def test_text_with_nowhere_to_insert_raises(self):
        with self.assertRaises(RuntimeError) as cm:
            h.hoist_imports("not spir-v\n")
        self.assertIn("couldn't find a place to insert imports", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
