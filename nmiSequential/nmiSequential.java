import java.util.*;

public class nmiSequential {
    public static void main(String[] args) {
        double[] gen1 = { -0.123983f, 0.06974f, -0.178225f, 0.023247f, -0.030996f, -0.092987f, -0.21697f, -0.340952f,
                -0.06974f, 0.092987f, 0.046493f, -0.030996f, 0.240216f, 0.457186f, 0.263463f, -0.21697f, -0.06974f,
                -0.340952f, 0.240216f, 0.023247f, 0.441688f, 0f, 0.015498f, -0.26f };
        double[] gen2 = { -0.253347f, -0.150639f, -0.109556f, 0.027389f, 0.362903f, -0.171181f, 0.054778f, -0.164333f,
                0.253347f, -0.150639f, 0.109556f, -0.143792f, 0.020542f, 0.020542f, 0.328667f, 0f, 0.150639f,
                -0.041083f, 0.054778f, 0.383445f, 0.328667f, -0.32182f, -0.308125f, -0.41f };

        System.out.println("NMI: " + calculationNMI(gen1, gen2, gen1.length));
    }

    public static float calculationNMI(double[] gen1, double[] gen2, int size) {
        float value = 0.0F;

        // Normalized arrays
        int[] gen1Normalized = new int[gen1.length];
        int[] gen2Normalized = new int[gen2.length];
        int maxVal = normalizedArray(gen1, gen1Normalized, gen1.length);
        normalizedArray(gen2, gen2Normalized, gen2.length);

        try {
            value = 2.0F * (float) calculateMutualInformation(gen1Normalized, gen2Normalized, size, maxVal)
                    / ((float) calculateEntropy(gen1, gen1Normalized, size) + (float) calculateEntropy(gen2, gen2Normalized, size));
        } catch (Exception e) {
            System.out.println("Error");
            value = 0.0F;
        }
        return value;
    }

    public static double calculateMutualInformation(int[] gen1, int[] gen2, int size, int maxVal) {
        double LOG_BASE = 2.0D;

        double[] probMap = new double[8]; // 2 (gen1) + 2 (gen2) + 4 (joint)

        for (int iColumn = 0; iColumn < size; ++iColumn) {
            int valGen1Column = gen1[iColumn];
            int valGen2Column = gen2[iColumn];

            probMap[valGen1Column] = probMap[valGen1Column] + 1;
            probMap[valGen2Column + 2] = probMap[valGen2Column + 2] + 1;
            probMap[(valGen1Column + maxVal * valGen2Column) + 4] = probMap[(valGen1Column + maxVal * valGen2Column) + 4] + 1;
        }

        for(int iCont = 0; iCont < 8; iCont++){
            probMap[iCont] = probMap[iCont] / size;
        }

        double nMI = 0.0D;
        for (int iCont = 0; iCont < 4; iCont++) {
            if (probMap[iCont + 4] > 0.0D && probMap[iCont / maxVal] > 0.0D && probMap[(iCont / maxVal)+2] > 0.0D) {
                nMI += probMap[iCont + 4]
                        * Math.log(probMap[iCont + 4] / probMap[iCont / maxVal] / probMap[(iCont / maxVal)+2]);
            }
        }

        nMI /= Math.log(LOG_BASE);
        return nMI;
    }

    public static final int normalizedArray(double[] gen, int[] genNormalized, int size) {
        int maxValue = 0;
        if (size > 0) {
            int minValue = (int) Math.floor(gen[0]);
            maxValue = (int) Math.floor(gen[0]);

            for (int iCont = 0; iCont < size; ++iCont) {
                int iExp = (int) Math.floor(gen[iCont]);
                genNormalized[iCont] = iExp;
                if (iExp < minValue) {
                    minValue = iExp;
                }

                if (iExp > maxValue) {
                    maxValue = iExp;
                }
            }

            for (int iCont = 0; iCont < size; ++iCont) {
                genNormalized[iCont] -= minValue;
            }

            maxValue = maxValue - minValue + 1;
        }

        return maxValue;
    }

    public static double calculateEntropy(double[] gen, int[] genNormalized, int size) {
        double LOG_BASE = 2.0D;

        double[] probMap = new double[2];

        for (int iColumn = 0; iColumn < size; ++iColumn) {
            int iExpr = genNormalized[iColumn];
            probMap[iExpr] = probMap[iExpr] + 1;
        }

        for(int iCont = 0; iCont < 2; iCont++){
            probMap[iCont] = probMap[iCont] / size;
        }

        double dEntropy = 0.0D;
        for(int iCont = 0; iCont < 2; iCont++){
            Double varAux = probMap[iCont];
            if(varAux > 0.0D){
                dEntropy -= varAux * Math.log(varAux);
            }
        }

        dEntropy /= Math.log(LOG_BASE);
        return dEntropy;
    }
}
