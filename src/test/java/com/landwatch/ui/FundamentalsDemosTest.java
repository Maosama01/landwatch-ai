package com.landwatch.ui;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.PrintStream;

import static org.junit.jupiter.api.Assertions.*;

/** The NN Fundamentals page runs the real, seeded computations and returns their output as text. */
class FundamentalsDemosTest {

    @Test
    @DisplayName("XOR: seeded 2-4-1 MLP learns all four patterns, loss reported from the real training run")
    void xor() {
        String out = FundamentalsDemos.runXor();
        assertTrue(out.contains("Patterns classified correctly at threshold 0.5: 4 / 4"), out);
        assertTrue(out.contains("Initial loss :"));
        assertTrue(out.contains("Final loss   :"));
        assertTrue(out.contains("epoch 6000:"));
        assertFalse(out.contains("NaN"));
        assertEquals(out.replaceAll("Wall time: \\d+ ms", ""), FundamentalsDemos.runXor().replaceAll("Wall time: \\d+ ms", ""),
                "seeded, so identical apart from the wall time");
    }

    @Test
    @DisplayName("Gradient check: all 13 parameters of the 2-3-1 network pass")
    void gradientCheck() {
        String out = FundamentalsDemos.runGradientCheck();
        assertTrue(out.contains("Status: PASSED"), out);
        assertTrue(out.contains("Total Parameters Checked: 13"), out);
        assertTrue(out.contains("Failed Parameters: 0"), out);
    }

    @Test
    @DisplayName("Learning-rules demo output is captured as a String and System.out is restored")
    void learningRules() {
        PrintStream before = System.out;
        String out = FundamentalsDemos.runLearningRules();
        assertSame(before, System.out, "System.out must be restored");
        assertTrue(out.contains("[1] HEBBIAN LEARNING"));
        assertTrue(out.contains("[5] BOLTZMANN LEARNING"));
        assertTrue(out.contains("Demo complete."));
    }
}
