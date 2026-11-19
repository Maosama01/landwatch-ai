package com.landwatch.ui;

import javafx.beans.property.DoubleProperty;
import javafx.beans.property.SimpleDoubleProperty;
import javafx.scene.Cursor;
import javafx.scene.control.Label;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.KeyCode;
import javafx.scene.layout.Region;
import javafx.scene.layout.StackPane;
import javafx.scene.shape.Rectangle;
import javafx.scene.shape.SVGPath;

/**
 * Before/after swipe comparison: the after image fills the frame and the before image is revealed to the
 * left of a draggable divider, so the same place can be compared at the two dates pixel by pixel.
 *
 * <p>Drag (or click) anywhere on the image to move the divider; with keyboard focus, the left and right
 * arrow keys move it by 2 %. The two images must have the same size, as for change detection.</p>
 */
public final class SwipeCompareView extends Region {

    private static final double PAD = 8;
    private static final double HANDLE = 30;
    private static final double KEY_STEP = 0.02;

    private final ImageView beforeView = new ImageView();
    private final ImageView afterView = new ImageView();
    private final Rectangle beforeClip = new Rectangle();
    private final Region divider = new Region();
    private final StackPane handle = new StackPane();
    private final Label beforeTag = new Label("Before");
    private final Label afterTag = new Label("After");
    private final DoubleProperty position = new SimpleDoubleProperty(this, "position", 0.5);

    private double imageLeft;
    private double imageWidth;

    public SwipeCompareView() {
        getStyleClass().add("swipe-compare");
        for (ImageView v : new ImageView[] {beforeView, afterView}) {
            v.setPreserveRatio(true);
            v.setSmooth(true);
            v.setManaged(false);
        }
        beforeView.setClip(beforeClip);

        divider.getStyleClass().add("swipe-divider");
        divider.setManaged(false);
        divider.setMouseTransparent(true);

        SVGPath glyph = new SVGPath();
        glyph.setContent("M-3 -4L-7 0L-3 4 M3 -4L7 0L3 4");
        glyph.getStyleClass().add("swipe-handle-glyph");
        handle.getChildren().add(glyph);
        handle.getStyleClass().add("swipe-handle");
        handle.setManaged(false);
        handle.setMouseTransparent(true);

        for (Label tag : new Label[] {beforeTag, afterTag}) {
            tag.getStyleClass().add("swipe-tag");
            tag.setManaged(false);
            tag.setMouseTransparent(true);
        }

        getChildren().addAll(afterView, beforeView, divider, handle, beforeTag, afterTag);
        position.addListener(o -> requestLayout());

        setCursor(Cursor.H_RESIZE);
        setFocusTraversable(true);
        setOnMousePressed(e -> {
            requestFocus();
            position.set(positionAt(e.getX(), imageLeft, imageWidth));
        });
        setOnMouseDragged(e -> position.set(positionAt(e.getX(), imageLeft, imageWidth)));
        setOnKeyPressed(e -> {
            if (e.getCode() == KeyCode.LEFT) {
                position.set(Math.max(0, position.get() - KEY_STEP));
                e.consume();
            } else if (e.getCode() == KeyCode.RIGHT) {
                position.set(Math.min(1, position.get() + KEY_STEP));
                e.consume();
            }
        });
    }

    /** Shows {@code before} left of the divider and {@code after} right of it; null clears the view. */
    public void setImages(Image before, Image after) {
        beforeView.setImage(before);
        afterView.setImage(after);
        requestLayout();
    }

    public boolean hasImages() {
        return beforeView.getImage() != null && afterView.getImage() != null;
    }

    /** Divider position from 0 (all after) to 1 (all before). */
    public DoubleProperty positionProperty() {
        return position;
    }

    /** See {@link ImageFit#fit}. */
    public static double[] fit(double imgW, double imgH, double boxW, double boxH) {
        return ImageFit.fit(imgW, imgH, boxW, boxH);
    }

    /** See {@link ImageFit#positionAt}. */
    public static double positionAt(double x, double left, double width) {
        return ImageFit.positionAt(x, left, width);
    }

    @Override
    protected void layoutChildren() {
        Image image = afterView.getImage();
        boolean show = image != null && beforeView.getImage() != null;
        for (javafx.scene.Node n : getChildren()) {
            n.setVisible(show);
        }
        if (!show) {
            return;
        }
        double[] size = fit(image.getWidth(), image.getHeight(), getWidth() - 2 * PAD, getHeight() - 2 * PAD);
        double w = size[0];
        double h = size[1];
        double x0 = (getWidth() - w) / 2;
        double y0 = (getHeight() - h) / 2;
        imageLeft = x0;
        imageWidth = w;

        for (ImageView v : new ImageView[] {beforeView, afterView}) {
            v.setFitWidth(w);
            v.setFitHeight(h);
            v.relocate(x0, y0);
        }
        double split = w * position.get();
        beforeClip.setWidth(split);
        beforeClip.setHeight(h);

        double dx = x0 + split;
        divider.resizeRelocate(dx - 1, y0, 2, h);
        handle.resizeRelocate(dx - HANDLE / 2, y0 + h / 2 - HANDLE / 2, HANDLE, HANDLE);

        double bw = beforeTag.prefWidth(-1);
        double bh = beforeTag.prefHeight(-1);
        beforeTag.resizeRelocate(x0 + 10, y0 + 10, bw, bh);
        double aw = afterTag.prefWidth(-1);
        afterTag.resizeRelocate(x0 + w - 10 - aw, y0 + 10, aw, afterTag.prefHeight(-1));
        beforeTag.setOpacity(position.get() > 0.12 ? 1 : 0);
        afterTag.setOpacity(position.get() < 0.88 ? 1 : 0);
    }
}
