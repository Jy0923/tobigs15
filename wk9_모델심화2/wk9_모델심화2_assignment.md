# Attention Is All You Need

논문 링크 : [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)



### Abstract

좋은 성능의 sequence transduction 모델은 encoder와 decoder를 포함하는 복잡한 recurrent, convolutional network를 기반으로 한다. 또한 최적의 성능을 내는 모델은 attention mechanism을 통해 encoder와 decoder를 연결한다. 이 논문은 recurrence와 convolution을 완전히 제거하고 전적으로 attention mechanism에 기반한 Transformer라는 단순한 모델 구조를 제시한다. 이 모델은 두개의 기계번역 task에서 우수한 성능을 내면서도 병렬 처리가 가능하고 학습시간은 훨씬 단축된다.



## 1 Introduction

RNN, LSTM, GRU는 특히 language modeling과 기계번역과 같은 sequence modeling과 transduction 문제에서 sota로 자리잡고 있으며, recurrent language model과 encoder-decoder구조의 사용을 넓히기 위해 많은 노력이 계속되었다.

Recurrent model은 일반적으로 입력 및 출력 sequence의 symbol position에 따라 계산한다. 계산 단계에서 위치에 따라 이전의 hidden state $h_{t-1}$와 위치 t를 입력으로 받아 $h_t$를 생성한다. 이러한 순차적인 특성은 긴 sequence의 처리에서 중요한 병렬화를 힘들게 한다. 

attention mechanism은 다양한 작업에서 sequence modeling 및 transduction model의 필수적인 부분이 되었으며, 입력 또는 출력 sequence들의 거리에 관계없이  dependency를 modeling할 수 있다. 하지만 거의 모든 경우에서 attention mechanism은 recurrent network와 함께 사용된다.

이 논문에서는 recurrence를 피하는 대신에 전적으로 attention mechanism에 의존하여 입력과 출력 사이의 global dependency를 그리는 모델인 Transformer를 제안한다.



## 2 Background

Extended Nearal GPU, ByteNet, ConvS2S는 순차적인 계산을 줄이는 목표를 가졌으며  모두 convolutional nearal network를 사용하여 모든 입력과 출력 position에 대해 병렬로 hidden representation을 계산한다. 이 모델들에서는 두 임의의 입력 또는 출력의 position에서 오는 신호를 연결하는데 필요한 연산의 수는 위치 간 거리에 대해 증가하는데, ConvS2S는 선형적으로, ByteNet은 대수적으로 증가하고, 이로 인해 먼 위치 간의 dependency를 학습하기 더 어려워진다. Transformer는 multi-head attention의 효과 덕분에 연산량이 constant하게 줄어든다.

intra-attention이라고도 불리는 self-attention은 sequence의 representation을 계산하기 위해 단일 sequence의 다른 위치를 연관시키는 attention mechanism이다. self-attention은 reading comprehension, abstractive summarization등의 다양한 작업에서 성공적으로 사용되었다.

그러나 Transformer는 정렬된 sequence의 RNN, convolution을 사용하지 않고 전적으로 self-attention에 의존하면서 입력과 출력의 representation을 계산하는 첫번째 transduction 모델이다.



## 3 Model Architecture

대부분의 경쟁력있는 nearal sequence transduction model은 encoder-decoder구조를 가지고 있다. encoder는 입력 sequence의 symbol representation $(x_1, ..., x_n)$을 연속적인 representation sequence $z = (z_1, ..., z_n)$에 mapping한다. 주어진 $z$에 대해  decoder는 한 번에 하나의 element씩 출력 sequence symbol $(y_1, ..., y_n)$을 생성한다. 각 step에서 모델은 auto-regressive하며 다음 step을 생성할 때 이전에 생성된 symbol을 추가적인 입력으로 사용한다.

Transformer는 Figure1에 표시된 encoder, decoder 모두에 대해 self-attention과 point-wise fully connected layer가 쌓여진 구조를 사용한다.

![](https://github.com/Jy0923/tobigs15/blob/master/wk9_%EB%AA%A8%EB%8D%B8%EC%8B%AC%ED%99%942/Figure1_Transformer.PNG)



#### 3.1 Encoder and Decoder Stacks

* Encoder : $N = 6$개의 동일한 layer stack으로 구성된다. 각 layer에는 두 개의 sub-layer가 있다. 

  * Multi-head self attention mechanism
  * position-wise fully connected feed-forward network이다.

  두 개의 하위 계층 각각에 residual-connection을 적용한 다음 layer normalization을 사용한다. 각 하위 계층의 출력은 LayerNorm($x$ + Sublayer($x$)) 이며 Sublayer($x$)는 하위계층 자체에 의해 구현된 함수이다. residual-connection을 용이하게 하기 위해 모델의 모든 sub-layer와 embedding layer들은 동일하게 출력차원 $d_{model} = 512$의  출력을 생성한다.

* Decoder : $N = 6$개의 동일한 layer stack으로 구성된다. encoder layer에 있는 두 개의 sub-layer 외에도 decoder는 세번째 sub-layer를 삽입한다. 

  * encoder stack의 출력에 대해 multi-head attention을 수행하는 sub-layer

  encoder와 비슷하게 각 계층에 대해 residual connection을 수행한 다음 layer normalization을 수행한다. 또한 현재 위치보다 뒤에 있는 요소에 attention을 적용하지 못하도록 decoder stack의 self-attention sub-layer를 수정한다(masked multi-head attention). 이러한 masking은 position i에 대한 예측이 i보다 작은 위치에서 알려진 출력에만 의존할 수 있도록 한다.

  (encoder의 경우 모든 input을 동시에 집어넣어 병렬적으로 수행하지만, decoder를 같은 방법으로 수행할 경우 아직 문장에서 등장하지 않은 미래의 단어를 참고하게 되므로 self-attention을 수행할 때 이미 transduction을 수행한 출력들에 대해서만 attention을 계산한다.)



#### 3.2 Attention

Attention function은 query와 key-value쌍의 집합을 출력에 mapping하는 것으로 표현할 수 있다. 출력은 value들의 가중 합으로 계산되며 각각의 value에 해당되는 가중치는 query와 그에 해당하는 key에 대한 compatibility function에 의해 계산된다.

![](https://github.com/Jy0923/tobigs15/blob/master/wk9_%EB%AA%A8%EB%8D%B8%EC%8B%AC%ED%99%942/Scaled_Dot_Product_Attention.PNG)


#### 3.2.1 Scaled Dot-Product Attention

입력은 $d_k$차원의 query와 key, 그리고 $d_v$차원의 value로 구성된다. query와 모든 key들 간 dot product를 계산한 후 $\sqrt{d_k}$로 나누어 scaling 해주고 softmax함수를 적용하여 value에 대한 weight를 얻는다. 

실제로는 query 집합을 행렬 $Q$로 묶어 attention function을 동시에 계산하며 key와 value역시 행렬 $K$와 행렬 $V$로 묶는다. $Q, K, V$를 이용하여 attention을 계산하는 법은 다음과 같다.

* Attention($Q, K, V$) = softmax$(\frac{QK^T}{\sqrt{d_k}})V$

가장 일반적으로 사용되는 두 가지 attention function은 additive attention과 dot-product attention이다. 

* additive attention
  * 단일 hidden layer의 feed-forward network를 사용하여 compatibility function을 계산
  * 큰 $d_k$에 대해 스케일링 없이 dot-product attention보다 좋은 성능
* dot-product attention
  * 이론적 복잡성은 additive attention과 비슷하지만 실제로 고도로 최적화된 행렬 곱셈을 사용하여 구현하기 때문에 더 빠르고 공간효율적
  * 큰 $d_k$에 대해 dot-product의 크기가 커져서 softmax함수에서 극심하게 작은 기울기가 있는곳으로 가게된다. 이를 막기 위해 $\sqrt{d_k}$로 나누어 스케일링



#### 3.2.2 Multi-Head Attention

![](https://github.com/Jy0923/tobigs15/blob/master/wk9_%EB%AA%A8%EB%8D%B8%EC%8B%AC%ED%99%942/Multi_Head_Attention.PNG)

$d_{model}$차원의 key, value, query를 사용하여 하나의 attention function을 수행하는 대신 학습된 서로 다른 linear projection을 사용하여 query, key, value를 각각 $d_k, d_k, d_v$차원으로 linearly projection을 $h$번 수행하는 것이 더 효과적이다.($d_k = d_v = d_{model}/h$)($h$번 linear projection 후 concat하여 $d_{model}$차원을 만들어준다) 각각 projected된 query, key, value들에 대해 attention function을 병렬로 수행하고 $d_v$차원의 출력값을 만든다. 이것들은 concat되고 다시 한번 projected되어 최종 값이 생성된다.

Multi-head attention을 사용하면 모델이 서로 다른 위치에 있는 서로 다른 representatin subspace의 정보에 공동으로 attention할 수 있는 반면에 단일의 attention head는 이것을 억제한다.

* MultiHead($Q, K, V$) = Concat($head_1, ..., head_h$)$W^O$
* $head_i$ = Attention($Q{W_i}^Q, K{W_i}^K, V{W_i}^V$)

이 논문에서 $h = 8$의 병렬 attention layer를 사용했고 각각의 layer에서 우리는 $d_k = d_v = d_{model}/h = 64$를 사용했다.(단어의 embedding차원 512, 512 / 8 = 64) 각 head에서 사용하는 dimension이 줄었기 때문에 전체 연산 비용은 single-head attention과 비슷하다.



#### 3.2.3 Applications of Attention in our Model

Transformer는 multi-head attention을 세가지 다른 방법으로 사용한다.

* encoder-decoder attention layer : query는 이전의 decoder layer에서 가져오고 key와 value는 encoder의 최종 출력에서 가져온다. 이는 decoder의 모든 위치가 입력 sequence의 모든 위치에 집중할 수 있게 한다.
* self-attention layer in encoder : encoder의 self-attention layer에서는 모든 key, value, query를 encoder의 이전 layer의 출력에서 가져온다. encoder의 각 position에서 encoder의 이전 layer에 대한 모든 position에 대해 attention을 적용할 수 있다.
* self-attention layer in decoder : decoder의 self-attention layer에서는 auto-regressive property(이전까지의 단어를 이용하여 뒤의 단어를 예측)를 유지하기 위해 decoder에서 정보가 왼쪽으로 흐르는 것을 방지해야 한다. 이를 방지하기 위해 scaled dot-product attentin에 masking out을 사용했다. i번째 position의 attention을 계산할 때 i+1번째 이후의 모든 position은 softmax함수의 input을 모두 $-\infty$로 설정하여 attention이 0이 되도록 한다.



#### 3.3 Position-wise Feed-Forward Networks

attention sub-layer외에도 encoder와 decoder의 각 계층에는 각 position에 개별적으로 동일하게 적용되는 fully connected feed-forward network가 있다. 이것은 두개의 linear transformation과 그 사이 하나의 relu로 이루어져 있다. 

* FFN($x$) = max(0, $xW_1 + b_1)W_2 + b_2$

linear transformation은 다른 position에 대해서 동일하게 적용되지만 layer마다 다른 parameter를 사용한다. 이는 커널 크기가 1인 convolution이다. 입력과 출력의 차원은 $d_{model} = 512$이고 inner-layer의 차원은 $d_{ff} = 2048$이다.



#### 3.4 Embeddings and Softmax

다른 sequence transduction model과 비슷하게 입력 token과 출력 token을 $d_{model}$차원의 벡터로 변환하기 위해 embedding을 학습한다. 일반적으로 학습된 linear transformatin과 softmax function을 사용하여 decoder의 출력을 예측된 다음 token의 확률로 변환한다. embedding layer와 softmax이전의 linear transformation에서 동일한 가중치행렬을 공유한다. embedding layer에서는 가중치에 $\sqrt{d_{model}}$을 곱한다.



#### 3.5 Positional Encoding

Transformer 모델이 recurrence와 convolution을 포함하지 않으므로 model이 sequence의 순서를 사용하려면 sequence에서 token의 상대적 또는 절대적인 position 정보를 주입해야한다. 이를 위해 encoder와 decoder stack의 하단에 있는 input embedding에 positional encoding을 추가한다. positional encoding은 embedding과 같이 $d_{model}$과 같은 차원을 가지며 둘을 더할 수 있으며 학습되거나 고정할 수 있다.

이 논문에서는 다른 주기를 갖는 sine함수와 cosine함수를 사용했다.

* $PE(pos, 2i) = sin(pos / 1000^{2i/d_{model}})$
* $PE(pos, 2i+1) = cos(pos / 1000^{2i/d_{model}})$

pos는 position을, i는 차원을 의미한다. positional encoding의 각 차원은 정현파에 해당하고, 파장은 $2\pi$에서 $10000 * 2\pi$로 기하학적 진행을 형성한다. 고정된 offset $k$에 대해 $PE_{pos+k}$함수는 $PE_{pos}$함수의 선형함수로 표현될 수 있기 때문에 모델이 상대적인 위치에 따라 attention하는 방법을 쉽게 배울 수 있다는 가설을 세웠고 이에 따라 sine함수와 cosine함수를 선택했다.

또한 학습된 positional embedding도 사용하여 실험해봤지만 두 결과가 거의 동일했다. 이 논문에서는 정현파를 선택했는데 그 이유는 학습중에 접하지 못한 더 긴 sequence에 대해 더 유연하게 대응할 수 있기 때문이다.



## 4 Why Self-Attention

4절에서는 self-attention layer와 한 가변길이 sequence의 representation $(x_1, ..., x_n)$을 동일한 길이의 다른 representation $(z_1, ..., z_n)$으로 mapping할 때 사용되는 recurrent, convolutional layer를 다양한 관점에서 비교한다. self-attention을 사용하도록 동기를 부여하기 위해 세가지 사항을 고려한다.

* layer 당 계산복잡도가 낮다.
* 병렬화 할 수 있는 계산의 양이 적다.(필요한 sequential operation의 수가 적다.)
* network에서 long-range dependency 간의 경로의 길이가 짧다.(long-range dependency를 학습하는 것은 sequential transduction의 핵심 과제이다.) 이러한 dependency의 학습에 영향을 미치는 중요한 요소중 하나는 순전파 및 역전파의 신호가 network에서 통과해야 하는 경로의 길이이다. 입력 및 출력 sequence의 위치 조합 사이의 경로가 짧을수록 long-range dependency를 배우는 것은 더 쉽다.

![](https://github.com/Jy0923/tobigs15/blob/master/wk9_%EB%AA%A8%EB%8D%B8%EC%8B%AC%ED%99%942/Table1_Transformer.PNG)

* self-attention layer와 recurrent layer

  - self-attention layer는 모든 위치를 연결하는데 상수시간의 연속적인 operation이 필요한 반면 recurrent layer는 $O(n)$의 연속적인 operation이 필요하다.

  - 계산복잡도 측면에서 self-attention layer는 sequence의 길이 $n$이 representation 차원인 $d$보다 작을 경우(기계번역에 사용되는 sota model의 경우 일반적으로 $n<d$) recurrent layer보다 빠르다.

  - sequence의 길이 $n$이 매우 큰 경우 계산성능을 향상시키기 위해서 self-attention은 각 출력 위치를 중심으로 하는 $r$만큼의 근접한 position만 고려하도록 제한할 수 있다. 하지만 이런 제약을 걸 경우 최대 경로의 길이가 $O(n/r)$로 늘어난다.(계산상의 이점은 있지만 long-range dependency 측면에서 좋지 않다.)

* self-attention layer와 convolutional layer

  * kernel의 크기 k가 n보다 작은 단일 convolutional layer는 모든 쌍의 input과 output position의 조합을 연결하지 못한다. 모든 조합을 연결하기 위해서 dilated convolution을 사용하면 $O(log_k(n))$개의 convolutional layer가 필요하며 어떤 두 위치 사이에서 가장 긴 경로의 길이가 늘어나게된다.
  * convolutional layer는 일반적으로 recurrent layer보다 k배가 더 비싸지만 분리 가능한 convolution은 계산복잡도를 $O(k * n * d + n * d^2)$로 상당히 감소시킨다. 그러나 k=n을 사용하더라도 분리 가능한 convolution의 복잡성은 transformer의 방식인 self-attention layer와 point-wise feed-forward layer의 조합과 같다.

추가적인 이점으로 self-attention은 더 이해하기 쉬운 모델을 생성한다. 개별 attention head는 다른 작업을 수행하는 방법을 명확하게 배울 뿐만 아니라 많은 부분이 문장의 구문 및 의미 구조와 관련되게 동작하는 것으로 보인다.
