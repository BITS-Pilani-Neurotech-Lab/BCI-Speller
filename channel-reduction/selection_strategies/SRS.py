from time import strftime
import SelectionStrategy

class SRS(SelectionStrategy):
	def run(self, acc_metric, max_channels):
		channels = [set() for _ in range(self.NUM_CHANNELS)]
		best_acc = [-1] * self.NUM_CHANNELS

		for count in range(max_channels):
			prev_channels = set()
			if count > 0:
				prev_channels = channels[count - 1]

			for i in range(self.NUM_CHANNELS):
				if i in prev_channels:
					continue

				curr_channels = prev_channels.copy()
				curr_channels.add(i)

				acc = self.get_accuracy(curr_channels)
				if acc[acc_metric] > best_acc[count]:
					best_acc[count] = acc[acc_metric]
					channels[count] = curr_channels

			print(strftime('%l:%M%p'), f'Electrodes: {count + 1}; Accuracy: {best_acc[count] * 100:.2f}%', flush = True)
		return channels, best_acc
