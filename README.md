# URBAN ENVIRONMENT AUDIO CLASSIFICATION

The goal of this project is to build an audio classifier capable of recognising different sounds. The sounds we have used are ambient noises from an urban environment. Right now we have focused on 4 categories of sounds. This can prove to be a very effective tool for audio surviellance of the sorrounding. It's applications can vary from as simple as making a device to detect a dog's bark to as big as detection of gun shots/bombing or other terrorist activities. We tried plotting different types of spectrograms of samples from each class which inturn lead us to see what filters would be best suited for this classification. In the end we used Mel-frequency cepstral coefficients, Mel-scaled power spectrogram, Chromagram of a short-time Fourier transform, Octave-based spectral contrast, Tonnetz to extract features. The coeffecients from these filters led to 193 features which were then used for classification using Machine Learning.					

## 1. Introduction
As mentioned, the goal of this project is to build an audio classifier capable of recognising different sounds that are heard in our environment. In this section, I would introduce you to the problem statement and the try to explain the challenges it posed, then present some litreature survey that we did and lastly explaining the approach that we decieded to take for solving this problem.

### 1.1 Introduction to Problem
The crux of the problem is this: given a never-before-heard recording, how can a system be trained to identify what it is actually listening to?
Let’s try to refine this problem. The input format will be the digital representation of the original analogue waveforms, just as would be recorded by a microphone, encoded using pulse-code modulation and stored as (lossless) WAV files.
From this input, we’ll need to extract features. But sound recordings are not spreadsheets, with their data neatly organised into rows and columns of known significance. Say we sample two recordings of equal duration, creating 2000 discrete floating point numbers; we can not meaningfully compare the 1000th number of one recording with the 1000th number of another. This is because we have no way of telling whether the number at any particular point in time is signal, silence or noise. Instead, we must abstract away from individual numbers, and consider data values in the context of many of its neighbours. This will grant us the ability to start spotting patterns which we call features, and permit us to generalise - the fundamental requirement in machine learning.
The challenge then, is to find measurable properties that differ in dissimilar recordings and are alike in those from similar sources. The complexity of feature extraction makes this problem an attractive application of deep learning, whose hierarchical nature makes it capable of automated feature learning. Once features have somehow been extracted, the question becomes how can we use them to train a model, and then use what we’ve learned to generate predictions?
Next, we’ll want to tune the model produced, to achieve the best possible accuracy. So we’ll need to consider how we can measure successful classification - and how can the performance of the model be optimised. Putting all this together suggests a processing pipeline as shown in Fig 1.

### 1.2 Figures

Fig. 1: An overview block diagram of the project.
![An overview block diagram of the project](https://ashutoshns.github.io/dsp-s5/img/fig1.jpg "Fig. 1: An overview block diagram of the project")

Fig 2: Amplitude vs. Time plot for sample files from all four categories.
![Amplitude vs. Time plot for sample files from all four categories](https://ashutoshns.github.io/dsp-s5/img/fig2.jpg "Fig 2: Amplitude vs. Time plot for sample files from all four categories.")

Fig 3: Spectrogram of sample files from all four categories.
![Spectrogram of sample files from all four categories](https://ashutoshns.github.io/dsp-s5/img/fig3.jpg "Fig 3: Spectrogram of sample files from all four categories.")

### 1.3 Literature Review
Write something here.
				<div class="subsection">
					<div class="headingss">1.4 Proposed Approach Idea</div>
					<div class="text">

						<!-- Start edit here  -->
						Describe your approach briefly here.
						<!-- Stop edit here -->

					</div>
				</div>
				<div class="subsection">
					<div class="headingss">1.5 Report Organization</div>
					<div class="text">

						<!-- Start edit here  -->
						Write something here.
						<!-- Stop edit here -->

					</div>
				</div>
			</div>

			<div class="section">
				<div class="heading">2. Proposed Approach</div>
				<div class="text">

					<!-- Start edit here  -->
					Write something here.
					<!-- Stop edit here -->

				</div>
			</div>

			<div class="section">
				<div class="heading">3. Experiments &amp; Results</div>
				<div class="subsection">
					<div class="headingss">3.1 Dataset Description</div>
					<div class="text">

						<!-- Start edit here  -->
						Write something here.
						<!-- Stop edit here -->

					</div>
				</div>
				<div class="subsection">
					<div class="headingss">3.2 Discussion</div>
					<div class="text">

						<!-- Start edit here  -->
						Write something here.
						<!-- Stop edit here -->

					</div>
				</div>
			</div>

			<div class="section">
				<div class="heading">4. Conclusions</div>
				<div class="subsection">
					<div class="headingss">4.1 Summary</div>
					<div class="text">

						<!-- Start edit here  -->
						Write something here.
						<!-- Stop edit here -->

					</div>
				</div>
				<div class="subsection">
					<div class="headingss">4.2 Future Extensions</div>
					<div class="text">

						<!-- Start edit here  -->
						Write something here.
						<!-- Stop edit here -->

					</div>
				</div>
			</div>
			
			
			
	</body>
</html>
