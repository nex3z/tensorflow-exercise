package com.nex3z.examples.androidtflitemnist;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;

public class TfLiteModel {
    private static final String LOG_TAG = TfLiteModel.class.getSimpleName();

    private static final String MODEL_PATH = "mnist.tflite";

    private static final int DIM_BATCH_SIZE = 1;
    private static final int DIM_IMG_SIZE_HEIGHT = 28;
    private static final int DIM_IMG_SIZE_WIDTH = 28;
    private static final int DIM_PIXEL_SIZE = 1;
    private static final int CATEGORY_COUNT = 10;

    private Interpreter mTfLite;
    private ByteBuffer mImgData = null;
    private int[] mImagePixels = new int[DIM_IMG_SIZE_HEIGHT * DIM_IMG_SIZE_WIDTH];
    private float[][] mResult = new float[1][CATEGORY_COUNT];

    TfLiteModel(Activity activity) throws IOException {
        mTfLite = new Interpreter(loadModelFile(activity));

        mImgData = ByteBuffer.allocateDirect(
                        4 * DIM_BATCH_SIZE * DIM_IMG_SIZE_HEIGHT * DIM_IMG_SIZE_WIDTH * DIM_PIXEL_SIZE);
        mImgData.order(ByteOrder.nativeOrder());
    }

    public float[] apply(Bitmap bitmap) {
        convertBitmapToByteBuffer(bitmap);
        mTfLite.run(mImgData, mResult);
        // ImageUtil.printImageArray(mProcessed[0], DIM_IMG_SIZE_HEIGHT, DIM_IMG_SIZE_WIDTH);
        return mResult[0];
    }

    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (mImgData == null) {
            return;
        }
        mImgData.rewind();

        bitmap.getPixels(mImagePixels, 0, bitmap.getWidth(), 0, 0,
                bitmap.getWidth(), bitmap.getHeight());

        int pixel = 0;
        for (int i = 0; i < DIM_IMG_SIZE_WIDTH; ++i) {
            for (int j = 0; j < DIM_IMG_SIZE_HEIGHT; ++j) {
                final int val = mImagePixels[pixel++];
                mImgData.putFloat(convertToGreyScale(val));
            }
        }
    }

    private float convertToGreyScale(int color) {
        return (((color >> 16) & 0xFF) + ((color >> 8) & 0xFF) + (color & 0xFF)) / 3.0f;
    }
}
