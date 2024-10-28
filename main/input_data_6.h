#include <tvmgen_default.h>
const size_t input_len = 784;
const static __attribute__((aligned(16))) float input_data[] = {0.38823530077934265, 0.4431372582912445, 0.49803921580314636, 0.5215686559677124, 0.545098066329956, 0.5686274766921997, 0.5921568870544434, 0.6078431606292725, 0.615686297416687, 0.6549019813537598, 0.6470588445663452, 0.5686274766921997, 0.5607843399047852, 0.4431372582912445, 0.46666666865348816, 0.45098039507865906, 0.2705882489681244, 0.3176470696926117, 0.0117647061124444, -0.16078431904315948, -0.1921568661928177, -0.3803921639919281, -0.5921568870544434, -0.5686274766921997, -0.6549019813537598, -0.7176470756530762, 0.20000000298023224, 0.686274528503418, 0.35686275362968445, 0.3960784375667572, 0.4745098054409027, 0.49803921580314636, 0.5137255191802979, 0.529411792755127, 0.5137255191802979, 0.5686274766921997, 0.6078431606292725, 0.545098066329956, 0.6000000238418579, 0.5686274766921997, 0.3803921639919281, 0.2705882489681244, 0.4588235318660736, 0.40392157435417175, 0.2549019753932953, 0.4901960790157318, 0.3019607961177826, 0.23137255012989044, 0.09803921729326248, -0.07450980693101883, -0.37254902720451355, -0.545098066329956, -0.6941176652908325, -0.7647058963775635, -0.027450980618596077, 0.6470588445663452, 0.3019607961177826, 0.3490196168422699, 0.364705890417099, 0.4588235318660736, 0.45098039507865906, 0.46666666865348816, 0.49803921580314636, 0.5372549295425415, 0.5372549295425415, 0.38823530077934265, 0.5215686559677124, 0.1764705926179886, 0.16078431904315948, 0.239215686917305, 0.3960784375667572, 0.46666666865348816, 0.16078431904315948, 0.2549019753932953, 0.30980393290519714, 0.32549020648002625, 0.37254902720451355, 0.2549019753932953, -0.09803921729326248, -0.1921568661928177, -0.38823530077934265, -0.545098066329956, -0.45098039507865906, 0.5607843399047852, 0.29411765933036804, 0.34117648005485535, 0.38823530077934265, 0.4274509847164154, 0.4117647111415863, 0.4274509847164154, 0.5058823823928833, 0.4588235318660736, 0.4901960790157318, 0.35686275362968445, 0.16078431904315948, -0.003921568859368563, 0.1764705926179886, 0.38823530077934265, 0.2705882489681244, 0.06666667014360428, 0.14509804546833038, 0.5686274766921997, 0.24705882370471954, 0.41960784792900085, 0.2549019753932953, 0.2235294133424759, 0.09019608050584793, 0.003921568859368563, -0.1921568661928177, -0.43529412150382996, -0.7098039388656616, 0.4745098054409027, 0.24705882370471954, 0.2862745225429535, 0.3019607961177826, 0.38823530077934265, 0.26274511218070984, 0.2235294133424759, 0.3019607961177826, 0.3176470696926117, 0.3176470696926117, 0.23137255012989044, 0.14509804546833038, 0.019607843831181526, 0.239215686917305, 0.1921568661928177, 0.12156862765550613, -0.10588235408067703, 0.10588235408067703, 0.4117647111415863, 0.32549020648002625, 0.1764705926179886, 0.27843138575553894, 0.2705882489681244, 0.11372549086809158, 0.05098039284348488, -0.12156862765550613, -0.2862745225429535, -0.5921568870544434, -0.03529411926865578, 0.26274511218070984, 0.2705882489681244, 0.2705882489681244, 0.29411765933036804, 0.10588235408067703, 0.14509804546833038, 0.2078431397676468, 0.3176470696926117, 0.16078431904315948, 0.15294118225574493, 0.2235294133424759, 0.18431372940540314, 0.1764705926179886, 0.4274509847164154, 0.27843138575553894, -0.04313725605607033, 0.2078431397676468, 0.3490196168422699, 0.2078431397676468, 0.16078431904315948, 0.24705882370471954, 0.27843138575553894, 0.06666667014360428, 0.03529411926865578, -0.05882352963089943, -0.20000000298023224, -0.4588235318660736, -0.6941176652908325, 0.239215686917305, 0.23137255012989044, 0.12156862765550613, 0.09019608050584793, 0.019607843831181526, 0.11372549086809158, 0.019607843831181526, 0.11372549086809158, 0.16862745583057404, 0.26274511218070984, 0.3803921639919281, 0.019607843831181526, 0.26274511218070984, 0.38823530077934265, 0.2078431397676468, 0.05882352963089943, 0.239215686917305, 0.10588235408067703, 0.12941177189350128, 0.2549019753932953, 0.1764705926179886, 0.06666667014360428, 0.18431372940540314, -0.03529411926865578, -0.04313725605607033, -0.15294118225574493, -0.4117647111415863, -0.7019608020782471, 0.2235294133424759, 0.239215686917305, 0.003921568859368563, -0.0117647061124444, -0.09019608050584793, -0.03529411926865578, -0.13725490868091583, 0.04313725605607033, 0.23137255012989044, 0.29411765933036804, 0.4431372582912445, -0.04313725605607033, 0.40392157435417175, 0.48235294222831726, 0.6392157077789307, 0.38823530077934265, 0.26274511218070984, 0.06666667014360428, 0.019607843831181526, 0.239215686917305, 0.1921568661928177, 0.2078431397676468, 0.2235294133424759, -0.23137255012989044, 0.04313725605607033, -0.12156862765550613, -0.3960784375667572, -0.6941176652908325, 0.14509804546833038, 0.05098039284348488, -0.239215686917305, -0.2862745225429535, -0.40392157435417175, -0.239215686917305, -0.11372549086809158, 0.12941177189350128, 0.07450980693101883, -0.11372549086809158, -0.239215686917305, -0.24705882370471954, -0.16862745583057404, -0.16078431904315948, -0.11372549086809158, 0.49803921580314636, 0.7019608020782471, 0.6627451181411743, 0.3490196168422699, 0.3333333432674408, 0.2235294133424759, 0.12941177189350128, -0.05882352963089943, -0.1764705926179886, -0.10588235408067703, -0.1921568661928177, -0.3019607961177826, -0.6627451181411743, 0.12941177189350128, 0.13725490868091583, -0.07450980693101883, -0.14509804546833038, -0.545098066329956, -0.3803921639919281, -0.239215686917305, -0.3333333432674408, -0.4901960790157318, -0.5686274766921997, -0.7333333492279053, -0.8588235378265381, -0.8117647171020508, -0.7176470756530762, -0.6313725709915161, -0.37254902720451355, -0.12156862765550613, -0.05882352963089943, 0.05098039284348488, 0.24705882370471954, 0.11372549086809158, 0.20000000298023224, 0.06666667014360428, -0.16862745583057404, 0.027450980618596077, -0.2078431397676468, -0.3803921639919281, -0.6627451181411743, 0.16078431904315948, -0.003921568859368563, -0.239215686917305, -0.35686275362968445, -0.45098039507865906, -0.5215686559677124, -0.5686274766921997, -0.6941176652908325, -0.8039215803146362, -0.8901960849761963, -0.8745098114013672, -0.8666666746139526, -0.8745098114013672, -0.8666666746139526, -0.8666666746139526, -0.8196078538894653, -0.7333333492279053, -0.5921568870544434, -0.34117648005485535, -0.2078431397676468, -0.05882352963089943, 0.12941177189350128, 0.15294118225574493, -0.10588235408067703, -0.09019608050584793, -0.239215686917305, -0.4431372582912445, -0.6784313917160034, 0.06666667014360428, -0.04313725605607033, -0.1764705926179886, -0.3176470696926117, -0.5372549295425415, -0.529411792755127, -0.6627451181411743, -0.8666666746139526, -0.9137254953384399, -0.8901960849761963, -0.8509804010391235, -0.8666666746139526, -0.8509804010391235, -0.8588235378265381, -0.843137264251709, -0.8666666746139526, -0.8823529481887817, -0.8745098114013672, -0.843137264251709, -0.6392157077789307, -0.41960784792900085, -0.21568627655506134, -0.18431372940540314, -0.18431372940540314, -0.11372549086809158, -0.40392157435417175, -0.34117648005485535, -0.686274528503418, -0.08235294371843338, -0.09019608050584793, -0.019607843831181526, -0.29411765933036804, -0.545098066329956, -0.29411765933036804, -0.5607843399047852, -0.772549033164978, -0.8980392217636108, -0.8901960849761963, -0.8745098114013672, -0.8588235378265381, -0.843137264251709, -0.8509804010391235, -0.8509804010391235, -0.8274509906768799, -0.8509804010391235, -0.8509804010391235, -0.8509804010391235, -0.8509804010391235, -0.8745098114013672, -0.772549033164978, -0.45098039507865906, -0.24705882370471954, -0.545098066329956, -0.5058823823928833, -0.5137255191802979, -0.7882353067398071, -0.03529411926865578, -0.14509804546833038, -0.3176470696926117, -0.3333333432674408, -0.05882352963089943, -0.09803921729326248, 0.03529411926865578, -0.34117648005485535, -0.8352941274642944, -0.8666666746139526, -0.8745098114013672, -0.8509804010391235, -0.843137264251709, -0.8352941274642944, -0.8196078538894653, -0.8352941274642944, -0.843137264251709, -0.843137264251709, -0.843137264251709, -0.8509804010391235, -0.843137264251709, -0.8352941274642944, -0.8823529481887817, -0.6705882549285889, -0.615686297416687, -0.6000000238418579, -0.545098066329956, -0.7960784435272217, -0.29411765933036804, -0.12941177189350128, -0.29411765933036804, -0.2705882489681244, -0.0117647061124444, 0.08235294371843338, -0.019607843831181526, 0.12941177189350128, -0.4745098054409027, -0.8980392217636108, -0.9058823585510254, -0.8588235378265381, -0.8117647171020508, -0.8352941274642944, -0.8196078538894653, -0.7960784435272217, -0.8196078538894653, -0.8117647171020508, -0.8039215803146362, -0.8117647171020508, -0.8352941274642944, -0.8666666746139526, -0.8666666746139526, -0.8666666746139526, -0.7490196228027344, -0.6627451181411743, -0.7019608020782471, -0.7176470756530762, -0.3019607961177826, -0.24705882370471954, -0.16862745583057404, -0.06666667014360428, 0.07450980693101883, 0.12156862765550613, -0.0117647061124444, 0.2235294133424759, 0.027450980618596077, -0.35686275362968445, -0.8901960849761963, -0.8901960849761963, -0.8823529481887817, -0.843137264251709, -0.8117647171020508, -0.7882353067398071, -0.7960784435272217, -0.7960784435272217, -0.8039215803146362, -0.8352941274642944, -0.8745098114013672, -0.8745098114013672, -0.8588235378265381, -0.8745098114013672, -0.8039215803146362, -0.7098039388656616, -0.6941176652908325, -0.7568627595901489, -0.2862745225429535, -0.239215686917305, -0.003921568859368563, 0.03529411926865578, 0.05882352963089943, 0.09019608050584793, 0.07450980693101883, 0.0117647061124444, 0.24705882370471954, 0.08235294371843338, -0.3803921639919281, -0.9137254953384399, -0.8666666746139526, -0.8352941274642944, -0.8352941274642944, -0.8196078538894653, -0.8352941274642944, -0.8509804010391235, -0.8745098114013672, -0.8666666746139526, -0.8666666746139526, -0.8823529481887817, -0.7960784435272217, -0.8745098114013672, -0.6470588445663452, -0.5137255191802979, -0.529411792755127, -0.7411764860153198, -0.239215686917305, -0.24705882370471954, -0.08235294371843338, 0.11372549086809158, 0.09803921729326248, 0.13725490868091583, 0.11372549086809158, -0.05882352963089943, 0.06666667014360428, 0.2862745225429535, 0.14509804546833038, 0.27843138575553894, -0.772549033164978, -0.8352941274642944, -0.7960784435272217, -0.7882353067398071, -0.8352941274642944, -0.843137264251709, -0.843137264251709, -0.8588235378265381, -0.8823529481887817, -0.843137264251709, -0.9215686321258545, -0.41960784792900085, -0.09803921729326248, -0.05098039284348488, -0.5215686559677124, -0.6627451181411743, -0.5372549295425415, -0.18431372940540314, 0.1764705926179886, 0.16078431904315948, 0.12156862765550613, 0.13725490868091583, 0.14509804546833038, -0.003921568859368563, -0.027450980618596077, 0.14509804546833038, 0.2549019753932953, 0.2549019753932953, 0.2862745225429535, 0.3960784375667572, 0.35686275362968445, -0.3490196168422699, -0.7490196228027344, -0.7960784435272217, -0.843137264251709, -0.8352941274642944, -0.8666666746139526, -0.8196078538894653, -0.3176470696926117, -0.019607843831181526, 0.13725490868091583, -0.12156862765550613, -0.41960784792900085, -0.6784313917160034, -0.4901960790157318, -0.4117647111415863, -0.23137255012989044, 0.0117647061124444, 0.09019608050584793, 0.1764705926179886, 0.10588235408067703, -0.019607843831181526, -0.1764705926179886, -0.03529411926865578, 0.12156862765550613, 0.2862745225429535, 0.4588235318660736, 0.5607843399047852, 0.3803921639919281, 0.38823530077934265, 0.364705890417099, 0.27843138575553894, -0.06666667014360428, -0.7176470756530762, -0.7098039388656616, -0.5764706134796143, 0.2705882489681244, 0.1764705926179886, 0.05098039284348488, -0.32549020648002625, -0.4117647111415863, -0.615686297416687, -0.48235294222831726, -0.32549020648002625, -0.38823530077934265, -0.10588235408067703, 0.0117647061124444, 0.12156862765550613, 0.12156862765550613, 0.04313725605607033, -0.2078431397676468, -0.18431372940540314, 0.0117647061124444, 0.23137255012989044, 0.38823530077934265, 0.4431372582912445, 0.2862745225429535, 0.3960784375667572, 0.38823530077934265, 0.49803921580314636, 0.37254902720451355, 0.26274511218070984, 0.09803921729326248, 0.2705882489681244, 0.30980393290519714, 0.15294118225574493, -0.09803921729326248, -0.45098039507865906, -0.43529412150382996, -0.5607843399047852, -0.5137255191802979, -0.35686275362968445, -0.4117647111415863, -0.07450980693101883, 0.05882352963089943, 0.12156862765550613, 0.11372549086809158, 0.10588235408067703, -0.027450980618596077, -0.3333333432674408, -0.5137255191802979, -0.06666667014360428, -0.003921568859368563, 0.239215686917305, 0.2705882489681244, 0.29411765933036804, 0.29411765933036804, 0.23137255012989044, 0.529411792755127, 0.40392157435417175, 0.34117648005485535, 0.5529412031173706, 0.04313725605607033, 0.12941177189350128, -0.40392157435417175, -0.49803921580314636, -0.4117647111415863, -0.5843137502670288, -0.5764706134796143, -0.5764706134796143, -0.46666666865348816, -0.2549019753932953, 0.04313725605607033, 0.09019608050584793, 0.05882352963089943, 0.03529411926865578, 0.14509804546833038, 0.03529411926865578, -0.3019607961177826, -0.5764706134796143, -0.5921568870544434, -0.364705890417099, -0.1764705926179886, 0.06666667014360428, 0.16862745583057404, 0.1921568661928177, 0.1921568661928177, 0.10588235408067703, -0.019607843831181526, -0.04313725605607033, -0.16862745583057404, -0.49803921580314636, -0.5921568870544434, -0.4431372582912445, -0.35686275362968445, -0.545098066329956, -0.5215686559677124, -0.6392157077789307, -0.4901960790157318, -0.34117648005485535, -0.05882352963089943, 0.09803921729326248, 0.07450980693101883, 0.003921568859368563, 0.027450980618596077, 0.12156862765550613, -0.027450980618596077, -0.37254902720451355, -0.5372549295425415, -0.6392157077789307, -0.7098039388656616, -0.5921568870544434, -0.4274509847164154, -0.14509804546833038, -0.10588235408067703, -0.15294118225574493, -0.24705882370471954, -0.364705890417099, -0.529411792755127, -0.615686297416687, -0.4745098054409027, -0.37254902720451355, -0.41960784792900085, -0.545098066329956, -0.6235294342041016, -0.5215686559677124, -0.4117647111415863, -0.38823530077934265, -0.07450980693101883, 0.05098039284348488, 0.09803921729326248, 0.0117647061124444, -0.05882352963089943, 0.0117647061124444, -0.0117647061124444, -0.2078431397676468, -0.37254902720451355, -0.5607843399047852, -0.6627451181411743, -0.7098039388656616, -0.7411764860153198, -0.6470588445663452, -0.5529412031173706, -0.529411792755127, -0.5686274766921997, -0.6549019813537598, -0.6235294342041016, -0.5137255191802979, -0.4117647111415863, -0.37254902720451355, -0.41960784792900085, -0.5372549295425415, -0.6000000238418579, -0.5921568870544434, -0.1921568661928177, -0.12941177189350128, -0.05098039284348488, -0.0117647061124444, 0.13725490868091583, -0.0117647061124444, -0.019607843831181526, -0.09019608050584793, 0.03529411926865578, 0.003921568859368563, 0.0117647061124444, -0.11372549086809158, -0.15294118225574493, -0.4117647111415863, -0.5843137502670288, -0.7333333492279053, -0.6549019813537598, -0.6392157077789307, -0.6235294342041016, -0.5843137502670288, -0.4431372582912445, -0.43529412150382996, -0.37254902720451355, -0.34117648005485535, -0.4274509847164154, -0.6392157077789307, -0.6549019813537598, -0.5058823823928833, -0.4745098054409027, -0.3176470696926117, -0.08235294371843338, 0.239215686917305, 0.26274511218070984, 0.21568627655506134, 0.03529411926865578, -0.09019608050584793, -0.0117647061124444, 0.21568627655506134, 0.27843138575553894, 0.2705882489681244, 0.239215686917305, 0.23137255012989044, 0.06666667014360428, -0.11372549086809158, -0.34117648005485535, -0.3490196168422699, -0.3960784375667572, -0.40392157435417175, -0.43529412150382996, -0.38823530077934265, -0.3333333432674408, -0.35686275362968445, -0.529411792755127, -0.7411764860153198, -0.6000000238418579, -0.6941176652908325, -0.6000000238418579, -0.29411765933036804, -0.29411765933036804, -0.09019608050584793, -0.04313725605607033, 0.23137255012989044, 0.2078431397676468, 0.06666667014360428, 0.09019608050584793, 0.1921568661928177, 0.23137255012989044, 0.2078431397676468, 0.26274511218070984, 0.239215686917305, 0.16078431904315948, 0.003921568859368563, -0.16078431904315948, -0.30980393290519714, -0.3803921639919281, -0.43529412150382996, -0.40392157435417175, -0.3333333432674408, -0.3960784375667572, -0.46666666865348816, -0.545098066329956, -0.8509804010391235};
