package com.landwatch.fundamentals.learningrules;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Smoke test: the presentation demo runs end-to-end and prints every section.
 */
class LearningRulesDemoTest {

    @Test
    @DisplayName("LearningRulesDemo.main runs without exceptions and prints all five rules")
    void testDemoRuns() {
        PrintStream original = System.out;
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        try {
            System.setOut(new PrintStream(buffer, true, StandardCharsets.UTF_8));
            LearningRulesDemo.main(new String[0]);
        } finally {
            System.setOut(original);
        }
        String output = buffer.toString(StandardCharsets.UTF_8);

        assertTrue(output.contains("[1] HEBBIAN LEARNING"));
        assertTrue(output.contains("[2] OJA'S RULE"));
        assertTrue(output.contains("[3] COMPETITIVE LEARNING"));
        assertTrue(output.contains("[4a] PERCEPTRON RULE"));
        assertTrue(output.contains("[4b] LMS"));
        assertTrue(output.contains("[5] BOLTZMANN LEARNING"));
        assertTrue(output.contains("Demo complete."));
        assertFalse(output.contains("NaN"), "Demo output must not contain NaN");
        assertFalse(output.contains("Infinity"), "Demo output must not contain Infinity");
    }
}
