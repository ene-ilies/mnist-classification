import softmax
import draw
import five_relu_and_softmax as rs

def drawCallback(img):
	global mnistModel
	print(mnistModel.fit(img))

#mnistModel = softmax.MNISTSoftmax()
#mnistModel.train()

mnistModel = rs.FiveReLUAndSoftmax()
mnistModel.train()

drawingWindow = draw.DrawingWindow(drawCallback)
drawingWindow.display()
