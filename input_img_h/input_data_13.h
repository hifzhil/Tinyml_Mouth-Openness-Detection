#include <tvmgen_default.h>
const size_t input_len = 784;
const static __attribute__((aligned(16))) float input_data[] = {0.13725490868091583, 0.15294118225574493, 0.16078431904315948, 0.16078431904315948, 0.16078431904315948, 0.16078431904315948, 0.13725490868091583, 0.12156862765550613, 0.08235294371843338, 0.05882352963089943, 0.05882352963089943, -0.027450980618596077, -0.10588235408067703, -0.16078431904315948, -0.27843138575553894, -0.38823530077934265, -0.45098039507865906, -0.43529412150382996, -0.529411792755127, -0.32549020648002625, -0.26274511218070984, -0.24705882370471954, -0.20000000298023224, -0.18431372940540314, -0.13725490868091583, -0.09019608050584793, -0.6235294342041016, -0.03529411926865578, 0.1764705926179886, 0.1764705926179886, 0.1764705926179886, 0.16862745583057404, 0.16078431904315948, 0.14509804546833038, 0.09019608050584793, -0.019607843831181526, -0.08235294371843338, -0.14509804546833038, -0.1921568661928177, -0.5921568870544434, -0.615686297416687, -0.6549019813537598, -0.7098039388656616, -0.529411792755127, -0.46666666865348816, -0.4745098054409027, -0.5764706134796143, -0.3176470696926117, -0.3333333432674408, -0.24705882370471954, -0.23137255012989044, -0.23137255012989044, -0.2235294133424759, -0.10588235408067703, 0.05098039284348488, -0.32549020648002625, 0.18431372940540314, 0.18431372940540314, 0.1764705926179886, 0.16862745583057404, 0.14509804546833038, 0.10588235408067703, -0.019607843831181526, -0.08235294371843338, -0.18431372940540314, -0.21568627655506134, -0.7960784435272217, -0.7647058963775635, -0.7490196228027344, -0.7647058963775635, -0.7411764860153198, -0.6705882549285889, -0.48235294222831726, -0.4431372582912445, -0.46666666865348816, -0.3803921639919281, -0.46666666865348816, -0.239215686917305, -0.20000000298023224, -0.1921568661928177, -0.21568627655506134, -0.12156862765550613, 0.05098039284348488, -0.40392157435417175, 0.29411765933036804, 0.2549019753932953, 0.239215686917305, 0.2078431397676468, 0.1764705926179886, 0.10588235408067703, -0.027450980618596077, -0.03529411926865578, -0.09019608050584793, -0.7882353067398071, -0.843137264251709, -0.8039215803146362, -0.6941176652908325, -0.615686297416687, -0.5137255191802979, -0.4431372582912445, -0.529411792755127, -0.49803921580314636, -0.41960784792900085, -0.3490196168422699, -0.30980393290519714, -0.23137255012989044, -0.16862745583057404, -0.14509804546833038, -0.1764705926179886, -0.16862745583057404, -0.04313725605607033, -0.05882352963089943, 0.9058823585510254, 0.9686274528503418, 0.9607843160629272, 0.9058823585510254, 0.9372549057006836, 0.24705882370471954, 0.11372549086809158, 0.03529411926865578, -0.6313725709915161, -0.7333333492279053, -0.7254902124404907, -0.6392157077789307, -0.4117647111415863, -0.30980393290519714, -0.2235294133424759, -0.12156862765550613, -0.12941177189350128, -0.3176470696926117, 0.41960784792900085, -0.545098066329956, -0.5529412031173706, 0.4588235318660736, 0.6313725709915161, 0.8196078538894653, 0.5372549295425415, 0.1921568661928177, 0.03529411926865578, -0.05098039284348488, 0.7254902124404907, 0.9764705896377563, 0.9529411792755127, 0.9607843160629272, 1.0, 0.9921568632125854, 1.0, 0.14509804546833038, -0.38823530077934265, -0.6784313917160034, -0.6549019813537598, -0.34117648005485535, -0.21568627655506134, -0.14509804546833038, -0.03529411926865578, 0.06666667014360428, 0.05098039284348488, 0.13725490868091583, 0.45098039507865906, 0.019607843831181526, -0.6627451181411743, 0.9058823585510254, 0.8745098114013672, 0.9686274528503418, 0.9529411792755127, 0.9764705896377563, 0.9450980424880981, 0.9764705896377563, 0.6000000238418579, 0.6627451181411743, 0.6549019813537598, 0.9843137264251709, 1.0, 1.0, 1.0, 0.21568627655506134, -0.4117647111415863, -0.6000000238418579, -0.5921568870544434, -0.11372549086809158, -0.027450980618596077, 0.08235294371843338, 0.16862745583057404, 0.239215686917305, 0.21568627655506134, 0.34117648005485535, 0.5372549295425415, -0.49803921580314636, -0.7647058963775635, -0.686274528503418, 0.8509804010391235, 0.9686274528503418, 0.9843137264251709, 0.9215686321258545, 0.9450980424880981, 0.4588235318660736, 0.615686297416687, 0.7333333492279053, 0.6549019813537598, 0.5764706134796143, 0.6392157077789307, 0.9921568632125854, 0.9529411792755127, -0.8588235378265381, -0.5137255191802979, -0.6549019813537598, -0.5686274766921997, -0.08235294371843338, 0.05882352963089943, 0.15294118225574493, 0.1921568661928177, 0.2705882489681244, 0.26274511218070984, 0.40392157435417175, 0.48235294222831726, 0.6313725709915161, -0.35686275362968445, -0.3490196168422699, 0.615686297416687, 0.9450980424880981, 0.9686274528503418, 0.5921568870544434, 0.9215686321258545, 0.9372549057006836, 0.6235294342041016, -0.7568627595901489, -0.772549033164978, -0.7960784435272217, -0.10588235408067703, -0.5686274766921997, -0.8588235378265381, -0.8823529481887817, -0.7254902124404907, -0.6470588445663452, -0.6549019813537598, -0.10588235408067703, 0.03529411926865578, 0.2235294133424759, 0.2705882489681244, 0.3019607961177826, 0.2862745225429535, 0.5686274766921997, -0.13725490868091583, 0.6392157077789307, -0.27843138575553894, -0.6470588445663452, 0.364705890417099, 0.9921568632125854, 0.9843137264251709, 0.8745098114013672, 0.9843137264251709, 0.4117647111415863, 0.615686297416687, -0.8509804010391235, -0.8745098114013672, -0.8745098114013672, -0.9137254953384399, -0.8980392217636108, -0.9137254953384399, -0.9137254953384399, -0.9058823585510254, -0.6313725709915161, -0.5686274766921997, -0.3019607961177826, 0.13725490868091583, 0.7490196228027344, 0.11372549086809158, 0.29411765933036804, 0.4274509847164154, 0.04313725605607033, -0.46666666865348816, 0.6784313917160034, -0.6078431606292725, -0.686274528503418, -0.6000000238418579, 0.9843137264251709, 0.9843137264251709, 0.6784313917160034, 0.4431372582912445, 0.7882353067398071, 0.027450980618596077, -0.8823529481887817, -0.9215686321258545, -0.9058823585510254, -0.9215686321258545, -0.8901960849761963, -0.9215686321258545, -0.9058823585510254, -0.09803921729326248, -0.23137255012989044, -0.37254902720451355, -0.6235294342041016, -0.019607843831181526, -0.24705882370471954, -0.003921568859368563, 0.529411792755127, 0.12941177189350128, 0.529411792755127, 0.13725490868091583, -0.16862745583057404, 0.7098039388656616, -0.08235294371843338, -0.29411765933036804, 1.0, 0.9764705896377563, 0.7882353067398071, 0.615686297416687, 0.7803921699523926, 0.6000000238418579, 0.14509804546833038, -0.9137254953384399, -0.929411768913269, -0.9215686321258545, -0.8980392217636108, -0.8980392217636108, -0.8823529481887817, -0.8039215803146362, 0.11372549086809158, -0.16862745583057404, -0.6784313917160034, -0.529411792755127, -0.6000000238418579, -0.8588235378265381, -0.10588235408067703, -0.8274509906768799, -0.34117648005485535, 0.929411768913269, -0.6470588445663452, 0.7019608020782471, -0.16078431904315948, -0.35686275362968445, 1.0, 1.0, 1.0, 1.0, 0.8274509906768799, -0.8745098114013672, -0.8980392217636108, -0.9137254953384399, -0.9215686321258545, -0.9215686321258545, -0.8980392217636108, -0.8823529481887817, -0.8823529481887817, -0.8588235378265381, -0.27843138575553894, -0.15294118225574493, 0.29411765933036804, -0.7490196228027344, -0.7647058963775635, -0.6470588445663452, -0.10588235408067703, 0.11372549086809158, -0.7333333492279053, 0.3490196168422699, -0.49803921580314636, 0.6705882549285889, -0.5921568870544434, -0.7098039388656616, 0.6313725709915161, 1.0, 1.0, 1.0, 1.0, -0.929411768913269, -0.9215686321258545, -0.9137254953384399, -0.9215686321258545, -0.8823529481887817, -0.8745098114013672, -0.8823529481887817, -0.8980392217636108, -0.8980392217636108, -0.06666667014360428, -0.09019608050584793, 0.07450980693101883, 0.1921568661928177, 0.16862745583057404, 0.13725490868091583, -0.05098039284348488, 0.3960784375667572, 0.6078431606292725, -0.686274528503418, -0.4745098054409027, 0.3333333432674408, 0.772549033164978, -0.615686297416687, -0.019607843831181526, 1.0, 1.0, 1.0, 1.0, -0.9607843160629272, -0.929411768913269, -0.9215686321258545, -0.9137254953384399, -0.8588235378265381, -0.8666666746139526, -0.8980392217636108, -0.9137254953384399, -0.9137254953384399, -0.9137254953384399, -0.2549019753932953, 0.08235294371843338, 0.30980393290519714, 0.40392157435417175, 0.0117647061124444, 0.1921568661928177, 0.5215686559677124, 0.6784313917160034, -0.5921568870544434, -0.45098039507865906, 0.3019607961177826, 0.772549033164978, -0.04313725605607033, -0.5137255191802979, 0.9921568632125854, 1.0, 1.0, 1.0, -0.9215686321258545, -0.9215686321258545, -0.9372549057006836, -0.9137254953384399, -0.8980392217636108, -0.9058823585510254, -0.9137254953384399, -0.929411768913269, -0.929411768913269, -0.929411768913269, -0.364705890417099, -0.0117647061124444, 0.2862745225429535, 0.2705882489681244, 0.09803921729326248, 0.2862745225429535, 0.4117647111415863, 0.30980393290519714, -0.4745098054409027, -0.48235294222831726, -0.027450980618596077, 0.7490196228027344, -0.41960784792900085, -0.09803921729326248, 0.9686274528503418, 0.9764705896377563, 1.0, 1.0, -0.9137254953384399, -0.9215686321258545, -0.9372549057006836, -0.9372549057006836, -0.9215686321258545, -0.929411768913269, -0.929411768913269, -0.9607843160629272, -0.9529411792755127, -0.9607843160629272, -0.46666666865348816, -0.03529411926865578, 0.24705882370471954, 0.09019608050584793, 0.12941177189350128, 0.12941177189350128, 0.5058823823928833, -0.364705890417099, -0.4117647111415863, -0.529411792755127, -0.26274511218070984, 0.7098039388656616, 0.6235294342041016, -0.6078431606292725, 0.545098066329956, 0.8588235378265381, 0.5372549295425415, 0.9607843160629272, -0.9607843160629272, -0.929411768913269, -0.9137254953384399, -0.9058823585510254, -0.9215686321258545, -0.9607843160629272, -0.9607843160629272, -0.9764705896377563, -0.9686274528503418, -0.7254902124404907, -0.29411765933036804, -0.003921568859368563, 0.03529411926865578, 0.27843138575553894, 0.06666667014360428, -0.2549019753932953, 0.4274509847164154, -0.364705890417099, -0.364705890417099, -0.4117647111415863, -0.3176470696926117, 0.6392157077789307, 0.7411764860153198, -0.6078431606292725, -0.43529412150382996, 0.843137264251709, 0.9058823585510254, 0.843137264251709, -0.9686274528503418, -0.9372549057006836, -0.8509804010391235, -0.9137254953384399, -0.9372549057006836, -0.9686274528503418, -0.6784313917160034, -0.5843137502670288, -0.7098039388656616, -0.6000000238418579, -0.16862745583057404, -0.14509804546833038, -0.03529411926865578, 0.24705882370471954, 0.09019608050584793, -0.14509804546833038, 0.32549020648002625, -0.34117648005485535, -0.2235294133424759, -0.5843137502670288, -0.37254902720451355, 0.5607843399047852, 0.6784313917160034, -0.16078431904315948, -0.4117647111415863, 0.8745098114013672, 0.8352941274642944, 0.8196078538894653, -0.9686274528503418, -0.9215686321258545, -0.929411768913269, -0.9450980424880981, -0.5529412031173706, -0.5843137502670288, -0.6941176652908325, -0.6549019813537598, -0.5607843399047852, -0.49803921580314636, -0.07450980693101883, -0.04313725605607033, -0.05882352963089943, 0.11372549086809158, 0.11372549086809158, 0.13725490868091583, -0.3960784375667572, -0.3176470696926117, -0.2235294133424759, -0.18431372940540314, -0.003921568859368563, 0.239215686917305, 0.6313725709915161, -0.3490196168422699, -0.3019607961177826, -0.2705882489681244, 0.8274509906768799, 0.8117647171020508, -0.9215686321258545, -0.9215686321258545, -0.929411768913269, -0.4274509847164154, -0.4588235318660736, -0.5215686559677124, -0.41960784792900085, -0.6470588445663452, -0.5921568870544434, -0.4745098054409027, -0.4745098054409027, 0.05882352963089943, -0.05098039284348488, 0.06666667014360428, 0.29411765933036804, 0.27843138575553894, -0.3019607961177826, -0.2078431397676468, -0.27843138575553894, -0.003921568859368563, 0.09803921729326248, -0.1764705926179886, 0.615686297416687, -0.49803921580314636, -0.23137255012989044, 0.3960784375667572, 0.3019607961177826, -0.03529411926865578, -0.929411768913269, -0.9372549057006836, -0.9215686321258545, -0.49803921580314636, -0.43529412150382996, -0.4901960790157318, -0.5921568870544434, -0.38823530077934265, -0.6078431606292725, -0.5215686559677124, -0.43529412150382996, -0.38823530077934265, 0.27843138575553894, 0.24705882370471954, 0.09803921729326248, -0.05098039284348488, -0.24705882370471954, -0.14509804546833038, -0.2862745225429535, -0.38823530077934265, -0.027450980618596077, 0.1764705926179886, 0.5215686559677124, 0.09803921729326248, -0.4117647111415863, -0.12156862765550613, 0.2235294133424759, 0.4117647111415863, -0.9215686321258545, -0.929411768913269, -0.41960784792900085, -0.48235294222831726, -0.5843137502670288, -0.38823530077934265, -0.41960784792900085, -0.5607843399047852, -0.35686275362968445, -0.6313725709915161, -0.3960784375667572, -0.37254902720451355, -0.32549020648002625, -0.3490196168422699, -0.16862745583057404, -0.1764705926179886, -0.16862745583057404, -0.05882352963089943, -0.16862745583057404, -0.26274511218070984, -0.09019608050584793, -0.05098039284348488, 0.2862745225429535, -0.2705882489681244, -0.4431372582912445, -0.34117648005485535, -0.4431372582912445, -0.2705882489681244, -0.929411768913269, -0.929411768913269, -0.41960784792900085, -0.48235294222831726, -0.4274509847164154, -0.4431372582912445, -0.4274509847164154, -0.27843138575553894, -0.40392157435417175, -0.34117648005485535, -0.48235294222831726, -0.3803921639919281, -0.2549019753932953, -0.20000000298023224, -0.1921568661928177, -0.08235294371843338, -0.08235294371843338, -0.04313725605607033, -0.21568627655506134, -0.18431372940540314, -0.15294118225574493, -0.2705882489681244, 0.18431372940540314, -0.4274509847164154, -0.5215686559677124, -0.5137255191802979, -0.6784313917160034, -0.6705882549285889, -0.929411768913269, -0.9215686321258545, -0.3803921639919281, -0.364705890417099, -0.5058823823928833, -0.5058823823928833, -0.3176470696926117, -0.3019607961177826, -0.3333333432674408, -0.21568627655506134, -0.2235294133424759, -0.1921568661928177, -0.2078431397676468, -0.35686275362968445, -0.24705882370471954, -0.15294118225574493, -0.07450980693101883, -0.03529411926865578, -0.3019607961177826, 0.9450980424880981, -0.1921568661928177, -0.1921568661928177, -0.05882352963089943, -0.48235294222831726, -0.7176470756530762, -0.7411764860153198, -0.7568627595901489, -0.7176470756530762, -0.929411768913269, -0.929411768913269, -0.3803921639919281, -0.43529412150382996, -0.4117647111415863, -0.34117648005485535, -0.48235294222831726, -0.45098039507865906, -0.29411765933036804, -0.239215686917305, -0.239215686917305, -0.2549019753932953, -0.4274509847164154, -0.46666666865348816, -0.2549019753932953, -0.08235294371843338, -0.04313725605607033, -0.239215686917305, -0.2078431397676468, 0.9372549057006836, -0.4588235318660736, -0.32549020648002625, -0.05882352963089943, 0.18431372940540314, -0.45098039507865906, -0.5921568870544434, -0.7882353067398071, -0.7411764860153198, -0.9215686321258545, -0.929411768913269, -0.4117647111415863, -0.40392157435417175, -0.4117647111415863, -0.32549020648002625, -0.3490196168422699, -0.2549019753932953, -0.3490196168422699, -0.2078431397676468, -0.239215686917305, 0.5843137502670288, 0.3803921639919281, -0.10588235408067703, -0.05098039284348488, -0.03529411926865578, 0.019607843831181526, -0.12941177189350128, 0.9215686321258545, 0.8901960849761963, -0.5529412031173706, -0.24705882370471954, -0.16078431904315948, 0.5607843399047852, -0.5372549295425415, -0.41960784792900085, -0.3803921639919281, -0.14509804546833038, -0.9215686321258545, -0.9215686321258545, -0.3960784375667572, -0.364705890417099, -0.545098066329956, -0.37254902720451355, -0.3176470696926117, -0.41960784792900085, -0.364705890417099, 0.8196078538894653, 0.8666666746139526, 0.8901960849761963, 0.9058823585510254, 0.8509804010391235, -0.027450980618596077, 0.0117647061124444, 0.07450980693101883, -0.05098039284348488, 0.8823529481887817, -0.09803921729326248, -0.15294118225574493, -0.3333333432674408, 0.6313725709915161, 0.7490196228027344, 0.8117647171020508, -0.46666666865348816, -0.41960784792900085, -0.3803921639919281};
