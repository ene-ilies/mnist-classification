import softmax
import draw
import five_relu_and_softmax as rs
import cnn_raw as cnnr

def drawCallback(img):
	global mnistModel
	print(mnistModel.fit(img))

#mnistModel = softmax.MNISTSoftmax()
#mnistModel.train()

#mnistModel = rs.FiveReLUAndSoftmax()
#mnistModel.train()

mnistModel = cnnr.CNNRawImplementation()
mnistModel.train()

drawingWindow = draw.DrawingWindow(drawCallback)
drawingWindow.display()
