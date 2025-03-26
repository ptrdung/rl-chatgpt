Tuyệt vời. Mình sẽ chuẩn bị một bài giảng chi tiết bằng tiếng Việt về phương pháp Value Function trong Reinforcement Learning. Bài sẽ bao gồm công thức, giải thích từng phần của công thức và ví dụ minh họa, tham khảo từ sách *Reinforcement Learning: An Introduction* (Sutton & Barto), *Mathematical Foundations of Reinforcement Learning* (Shiyu Zhao), và các nguồn uy tín khác.

Mình sẽ gửi lại bài giảng đầy đủ sau khi hoàn thành.

# Phương pháp Hàm Giá trị trong Học Tăng cường (Reinforcement Learning)

## Giới thiệu

Trong **Học tăng cường (Reinforcement Learning)**, một tác tử (agent) tương tác với môi trường thông qua các trạng thái và hành động, nhận được phần thưởng (reward) từ môi trường. Mục tiêu của tác tử là **tối đa hóa tổng phần thưởng tích lũy** theo thời gian. Để làm được điều này, tác tử thường ước lượng **hàm giá trị** – một hàm đánh giá “mức độ tốt” của một trạng thái hoặc của một hành động tại một trạng thái. Bài giảng này tập trung vào các phương pháp sử dụng hàm giá trị (Value Function) trong RL, bao gồm khái niệm hàm giá trị trạng thái $V(s)$ và hàm giá trị hành động $Q(s,a)$, các phương trình Bellman liên quan, cũng như các thuật toán học hàm giá trị tiêu biểu: **Monte Carlo**, **Temporal Difference** (TD – bao gồm TD(0) và TD($\lambda$)), **SARSA** và **Q-learning**. Mỗi phương pháp sẽ được trình bày với công thức cập nhật, giải thích chi tiết và ví dụ minh họa (như môi trường lưới – *Gridworld*), cùng với phân tích ưu nhược điểm.

## Hàm giá trị trạng thái $V(s)$ và hàm giá trị hành động $Q(s,a)$

**Hàm giá trị trạng thái $V(s)$** dưới một chính sách $\pi$ (ký hiệu $V^\pi(s)$) được định nghĩa là **giá trị kỳ vọng của tổng phần thưởng thu được** (thường được chiết khấu) khi tác tử bắt đầu ở trạng thái $s$ và **thực thi theo chính sách $\pi$** về sau ([Part 1: Key Concepts in RL — Spinning Up  documentation](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#:~:text=1.%20The%20On,pi)). Nói cách khác, $V^\pi(s)$ dự đoán mức phần thưởng mà tác tử sẽ nhận được trong tương lai nếu ở trạng thái $s$ và hành động theo $\pi$. Về mặt toán học, với $G_t$ là *return* (tổng phần thưởng chiết khấu từ thời điểm $t$ trở đi), ta có công thức định nghĩa: 

$$
V^\pi(s) \;=\; \mathbb{E}_\pi \big[\,G_t \mid S_t = s\,\big]\,,
$$

trong đó kỳ vọng được lấy theo chính sách $\pi$ khi bắt đầu từ trạng thái $s$ ([Part 1: Key Concepts in RL — Spinning Up  documentation](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#:~:text=1.%20The%20On,pi)). Thông thường, *return* được định nghĩa là $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots$, với $R_{t+1}$ là phần thưởng nhận được ngay sau trạng thái $s$ và $\gamma \in [0,1)$ là hệ số chiết khấu.

**Hàm giá trị hành động $Q(s,a)$** dưới chính sách $\pi$ (ký hiệu $Q^\pi(s,a)$) được định nghĩa tương tự, nhưng đánh giá giá trị của việc thực hiện *hành động $a$ tại trạng thái $s$*, sau đó tiếp tục theo chính sách $\pi$. Cụ thể, $Q^\pi(s,a)$ là **giá trị kỳ vọng của tổng phần thưởng** nếu tác tử ở trạng thái $s$, chọn hành động $a$, rồi từ bước tiếp theo trở đi hành động theo chính sách $\pi$ ([Part 1: Key Concepts in RL — Spinning Up  documentation](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#:~:text=2.%20The%20On,according%20to%20policy%20%20135)). Công thức định nghĩa:

$$
Q^\pi(s,a) \;=\; \mathbb{E}_\pi \big[\,G_t \mid S_t = s,\, A_t = a\,\big]\,.
$$

Hàm $Q^\pi(s,a)$ còn được gọi là **hàm giá trị trạng thái-hành động**. Mối quan hệ giữa $V$ và $Q$ là: giá trị trạng thái chính là kỳ vọng của giá trị hành động theo phân phối chọn hành động của chính sách: 
$$V^\pi(s) = \mathbb{E}_{a \sim \pi(\cdot|s)}[\,Q^\pi(s,a)\,]\,.$$ 
Ngược lại, nếu biết hàm $Q^*$ tối ưu, ta có thể tìm chính sách tối ưu bằng cách chọn hành động có $Q^*(s,a)$ lớn nhất ở mỗi trạng thái ([Part 1: Key Concepts in RL — Spinning Up  documentation](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#:~:text=Image%3A%20V)).

**Ví dụ:** Hình dưới đây minh họa giá trị trạng thái $V(s)$ trong một môi trường *Gridworld* 5×5 với chính sách chọn hành động ngẫu nhiên đều (mỗi hướng đi có xác suất bằng nhau). Mỗi ô biểu diễn giá trị $V(s)$ của trạng thái tương ứng; những giá trị dương cao hơn ở góc trên cho thấy từ các trạng thái đó kỳ vọng nhận được tổng thưởng lớn hơn (do gần mục tiêu hoặc có lối tắt đặc biệt) ([6 Reinforcement Learning.ppt](https://www.csd.uwo.ca/~dlizotte/teaching/slides/reinforcement_learning_1.pdf#:~:text=State,ball%20is%20in%20the%20hole)). 

 ([reinforcement learning - Gridworld from Sutton's RL book: how to calculate value function for corner cells? - Stack Overflow](https://stackoverflow.com/questions/64013919/gridworld-from-suttons-rl-book-how-to-calculate-value-function-for-corner-cell)) *Hàm giá trị trạng thái $V(s)$ cho chính sách random đều trong môi trường Gridworld (mỗi ô là một trạng thái, số trong ô là $V(s)$) ([6 Reinforcement Learning.ppt](https://www.csd.uwo.ca/~dlizotte/teaching/slides/reinforcement_learning_1.pdf#:~:text=State,ball%20is%20in%20the%20hole)).*

Trong ví dụ trên, các trạng thái có giá trị **3.3, 8.8** (góc trên) cao hơn hẳn so với các trạng thái xa (giá trị âm) phản ánh sự khác biệt về kỳ vọng phần thưởng tương lai. Chính sách ngẫu nhiên đều thường dẫn đến giá trị thấp ở các trạng thái biên (do bị phạt -1 khi đi ra khỏi lưới) và giá trị cao ở gần khu vực nhận thưởng đặc biệt (như trạng thái được teleport với phần thưởng dương).

## Phương trình Bellman cho hàm giá trị

Hàm giá trị thỏa mãn tính chất đệ quy được thể hiện qua **phương trình Bellman**. Ý tưởng cơ bản của phương trình Bellman: *“Giá trị của một trạng thái bằng phần thưởng kỳ vọng nhận được khi ở trạng thái đó, cộng với giá trị chiết khấu của trạng thái kế tiếp mà ta sẽ tới”* ([Part 1: Key Concepts in RL — Spinning Up  documentation](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#:~:text=,of%20wherever%20you%20land%20next)). 

Đối với **một chính sách $\pi$ bất kỳ**, *phương trình Bellman* cho hàm giá trị trạng thái $V^\pi(s)$ là:

$$
V^\pi(s) \;=\; \mathbb{E}_{a \sim \pi(\cdot|s),\, s' \sim P} \big[\, r(s,a) + \gamma \,V^\pi(s') \,\big]\,,
$$

trong đó $r(s,a)$ là phần thưởng nhận được khi thực hiện hành động $a$ ở trạng thái $s$, và $s' \sim P(\cdot|s,a)$ là trạng thái kế tiếp theo xác suất chuyển trạng thái của môi trường ([Part 1: Key Concepts in RL — Spinning Up  documentation](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#:~:text=The%20Bellman%20equations%20for%20the,policy%20value%20functions%20are)). Tương tự, phương trình Bellman cho hàm giá trị hành động là:

$$
Q^\pi(s,a) \;=\; \mathbb{E}_{s' \sim P} \big[\, r(s,a) + \gamma \,\mathbb{E}_{a' \sim \pi(\cdot|s')}\big[ Q^\pi(s',a')\big] \,\big]\,,
$$

nghĩa là giá trị của cặp trạng thái-hành động $(s,a)$ bằng phần thưởng thực hiện $a$ tại $s$ cộng với giá trị chiết khấu của hành động tiếp theo $a'$ tại trạng thái kế tiếp $s'$ (hành động $a'$ được chọn theo chính sách $\pi$) ([Part 1: Key Concepts in RL — Spinning Up  documentation](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#:~:text=The%20Bellman%20equations%20for%20the,policy%20value%20functions%20are)). Các phương trình Bellman này áp dụng cho mọi trạng thái (hoặc cặp trạng thái-hành động) – chúng tạo thành một hệ phương trình tuyến tính mà nghiệm chính là $V^\pi$ (hoặc $Q^\pi$).

Đặc biệt, với **hàm giá trị tối ưu** $V^*(s)$ và $Q^*(s,a)$ (ứng với chính sách tối ưu $\pi^*$), phương trình Bellman trở thành phương trình tối ưu (có thêm phép lấy max do tác tử luôn chọn hành động tốt nhất): 

$$
V^*(s) = \max_{a}\;\mathbb{E}_{s' \sim P}\big[\,r(s,a) + \gamma V^*(s')\,\big], \qquad 
Q^*(s,a) = \mathbb{E}_{s' \sim P}\big[\, r(s,a) + \gamma \max_{a'} Q^*(s',a') \,\big]\,.
$$

Sự khác biệt so với trường hợp chính sách cố định là ở chỗ xuất hiện $\max_{a'}$ – tác tử tối ưu luôn chọn hành động đem lại giá trị cao nhất ([Part 1: Key Concepts in RL — Spinning Up  documentation](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#:~:text=Image%3A%20%5Cbegin%7Balign,s%27%2Ca%27%29%7D.%20%5Cend%7Balign)). Phương trình Bellman tối ưu này chính là nền tảng cho thuật toán Q-learning mà chúng ta sẽ thảo luận sau.

## Phương pháp Monte Carlo trong học hàm giá trị

**Phương pháp Monte Carlo (MC)** ước lượng hàm giá trị bằng cách **sử dụng trung bình của các *return* thực nghiệm** thu được từ nhiều phiên tương tác (episodes). Không giống như phương pháp động (Dynamic Programming) đòi hỏi biết trước mô hình môi trường, Monte Carlo **không cần mô hình** mà học trực tiếp từ trải nghiệm mẫu – tức các chuỗi trạng thái, hành động, phần thưởng mà tác tử thu thập được ([Foundations of Reinforcement Learning  Model-free RL: Monte Carlo and temporal difference (TD) learning](https://users.ece.cmu.edu/~yuejiec/ece18813B_notes/lecture8-model-free-MC-TD.pdf#:~:text=Incremental%20Monte%20Carlo%20update%20The,z%20%7D%20incremental%20update)). 

**Nguyên tắc:** Monte Carlo quan sát phần thưởng tích lũy thực sự sau mỗi lần ghé thăm một trạng thái để cập nhật ước lượng giá trị. Cụ thể, giả sử tác tử chạy nhiều tập (episode) theo chính sách $\pi$. Mỗi khi trạng thái $s$ xuất hiện trong một tập và từ đó đến khi kết thúc tập thu được *return* $G$, thì $G$ được xem như một mẫu cho giá trị thực sự $v_\pi(s)$. Để ước lượng $V(s)$, ta có thể **lấy trung bình** các *return* đã quan sát mỗi lần trạng thái $s$ được ghé thăm (phương pháp MC đầu tiên – *first-visit* hoặc mỗi-lần – *every-visit*). Công thức cập nhật trung bình cho $V(s)$ sau $N(s)$ lần thăm là:

$$
V(s) \;\leftarrow\; V(s) + \frac{1}{N(s)}\Big(G - V(s)\Big)\,,
$$

tương đương với việc gán $V(s)$ bằng giá trị trung bình mới của các mẫu $G$ thu thập được ([Foundations of Reinforcement Learning  Model-free RL: Monte Carlo and temporal difference (TD) learning](https://users.ece.cmu.edu/~yuejiec/ece18813B_notes/lecture8-model-free-MC-TD.pdf#:~:text=Incremental%20Monte%20Carlo%20update%20The,z%20%7D%20incremental%20update)). Trong thực tế, thường người ta dùng **cập nhật gia số (incremental)** với hệ số học $\alpha$ cố định để có khả năng “quên dần” các kinh nghiệm cũ (giúp thích nghi khi môi trường thay đổi). Công thức dạng **constant-$\alpha$ MC** như sau:

$$
V(s) \;\leftarrow\; V(s) + \alpha\,\big(G - V(s)\big)\,,
$$

trong đó $\alpha \in (0,1]$ là tốc độ học (step size) ([Foundations of Reinforcement Learning  Model-free RL: Monte Carlo and temporal difference (TD) learning](https://users.ece.cmu.edu/~yuejiec/ece18813B_notes/lecture8-model-free-MC-TD.pdf#:~:text=This%20motivates%20a%20more%20general,10)). Bước cập nhật này điều chỉnh $V(s)$ tiến dần về *return* thực tế $G$ thu được.

**Ví dụ:** Xét một tác tử di chuyển ngẫu nhiên trong *Gridworld* cho đến khi vào ô đích và kết thúc episode. Giả sử với chính sách hiện tại, từ trạng thái $s$ tác tử đã chạy 2 episode độc lập và thu được tổng thưởng $G^{(1)} = -3$ và $G^{(2)} = -1$ (ví dụ do mỗi bước bị phạt -1 và mất 3 hoặc 1 bước để về đích). Khi đó ước lượng Monte Carlo cho $V(s)$ sẽ là trung bình $(-3 + -1)/2 = -2$. Nếu sử dụng cập nhật gia số với $\alpha=0.5$, xuất phát từ giá trị khởi tạo $V(s)=0$, ta sẽ cập nhật $V(s) = 0 + 0.5 * (-3 - 0) = -1.5$ sau episode đầu, rồi $V(s) = -1.5 + 0.5 * (-1 - (-1.5)) = -1.25$ sau episode thứ hai. Giá trị $V(s)$ dần hội tụ về -2 (giá trị thực tế dưới chính sách đó) khi số episode tăng lên.

**Ưu điểm:** Phương pháp Monte Carlo **đơn giản** và trực quan, không yêu cầu biết trước mô hình xác suất chuyển trạng thái của môi trường (model-free). Khi số lượng tập đủ lớn, ước lượng MC sẽ **không chệch** (unbiased) – về lý thuyết sẽ hội tụ về giá trị đúng $v_\pi(s)$. Monte Carlo cũng dễ triển khai song song do các tập độc lập với nhau.

**Nhược điểm:** Nhược điểm chính của MC là phải **chờ đến khi kết thúc episode** mới có số liệu để cập nhật. Điều này khiến việc học chậm nếu episode dài, và **không áp dụng được cho các nhiệm vụ không có trạng thái kết thúc** (continuing tasks) trừ khi ta cắt tập theo khoảng thời gian tùy ý ([Foundations of Reinforcement Learning  Model-free RL: Monte Carlo and temporal difference (TD) learning](https://users.ece.cmu.edu/~yuejiec/ece18813B_notes/lecture8-model-free-MC-TD.pdf#:~:text=Caveat%20of%20Monte%20Carlo%20methods%3A,but%20consistent%20under%20mild%20conditions)). Ngoài ra, ước lượng MC thường có **phương sai cao** – do *return* của mỗi tập phụ thuộc vào toàn bộ chuỗi tương tác ngẫu nhiên, nên rất biến động giữa các lần thử ([Foundations of Reinforcement Learning  Model-free RL: Monte Carlo and temporal difference (TD) learning](https://users.ece.cmu.edu/~yuejiec/ece18813B_notes/lecture8-model-free-MC-TD.pdf#:~:text=Caveat%20of%20Monte%20Carlo%20methods%3A,but%20consistent%20under%20mild%20conditions)) ([Lecture 4 - Model Free Techniques - MC and TD[Notes] - Omkar Ranadive](https://omkar-ranadive.github.io/posts/rl-l4-ds#:~:text=Therefore%3A)). Đổi lại, MC không dùng bootstrapping nên không bị thiên lệch: mỗi ước lượng $G$ thu được là “thật” từ môi trường (zero bias) ([Lecture 4 - Model Free Techniques - MC and TD[Notes] - Omkar Ranadive](https://omkar-ranadive.github.io/posts/rl-l4-ds#:~:text=As%20the%20returns%20in%20Monte,noise%2Foutlier%20obtained%20during%20the%20episodes)). Tóm lại, có thể nhớ: *Monte Carlo: phương sai cao, không chệch; TD: phương sai thấp, có chệch* ([Lecture 4 - Model Free Techniques - MC and TD[Notes] - Omkar Ranadive](https://omkar-ranadive.github.io/posts/rl-l4-ds#:~:text=Therefore%3A)).

## Phương pháp Temporal Difference (TD)

**Temporal Difference (TD)** là phương pháp học hàm giá trị kết hợp ý tưởng của Monte Carlo và phương pháp động (*bootstrapping*). Thuật toán TD **cập nhật giá trị sau mỗi bước** tương tác dựa trên **phần thưởng nhận được ngay** cộng với **ước lượng giá trị của trạng thái kế tiếp**, thay vì chờ đến cuối episode ([Monte Carlo vs Temporal Difference Learning - Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course/en/unit2/mc-vs-td#:~:text=To%20summarize%3A)). Nói cách khác, TD sử dụng *return* dự đoán một bước – thường gọi là **TD target** – để cập nhật giá trị, do đó thuộc loại *bootstrapping* (tương tự phương pháp giá trị hiện thời dự đoán cho chính nó).

### Phương pháp TD(0)

**TD(0)** (hay one-step TD) là trường hợp đơn giản nhất của TD, trong đó ta cập nhật giá trị trạng thái ngay sau mỗi bước thời gian (sử dụng thông tin của một bước tương lai). Công thức cập nhật cho **hàm giá trị trạng thái** theo TD(0) là:

$$ 
V(S_t) \;\leftarrow\; V(S_t) + \alpha\,\big[\,R_{t+1} + \gamma\,V(S_{t+1}) - V(S_t)\,\big]\,,
$$

trong đó $S_t$ là trạng thái hiện tại, $A_t$ là hành động thực hiện, $R_{t+1}$ là phần thưởng nhận được, $S_{t+1}$ là trạng thái kế tiếp; biểu thức trong dấu ngoặc vuông chính là **sai số TD**: 

$$
\delta_t = R_{t+1} + \gamma\,V(S_{t+1}) - V(S_t)\,. 
$$

Ta gọi $R_{t+1} + \gamma V(S_{t+1})$ là **TD target** (mục tiêu tạm thời) – nó kết hợp *thông tin mẫu* ($R_{t+1}$ nhận được) và *ước lượng hiện tại* ($V(S_{t+1})$) ([Foundations of Reinforcement Learning  Model-free RL: Monte Carlo and temporal difference (TD) learning](https://users.ece.cmu.edu/~yuejiec/ece18813B_notes/lecture8-model-free-MC-TD.pdf#:~:text=V%20,st%29%20%11)) ([Foundations of Reinforcement Learning  Model-free RL: Monte Carlo and temporal difference (TD) learning](https://users.ece.cmu.edu/~yuejiec/ece18813B_notes/lecture8-model-free-MC-TD.pdf#:~:text=%E2%80%A2%20TD%20target%20rt%20%2B,st%29%2014)). Thuật toán TD(0) điều chỉnh $V(S_t)$ một lượng tỉ lệ với sai số TD này. Nếu $\delta_t$ dương (nghĩa là *return* thực tế lớn hơn dự đoán), $V(S_t)$ sẽ được tăng lên; ngược lại nếu $\delta_t$ âm, $V(S_t)$ bị giảm xuống.

**Ví dụ:** Giả sử $\alpha = 0.1$ và $\gamma = 1$. Tại trạng thái $S$, hiện tại $V(S)=0$. Tác tử thực hiện một hành động và nhận được phần thưởng $R_{t+1}=+1$, chuyển đến trạng thái $S'$ với $V(S')$ ban đầu là 0. Khi đó TD(0) cập nhật: 

$$V(S) \leftarrow 0 + 0.1 * [\,1 + 1*0 - 0\,] = 0.1\,.$$ 

Như vậy giá trị $V(S)$ được tăng nhẹ từ 0 lên 0.1 dựa trên phần thưởng +1 vừa thấy ([Monte Carlo vs Temporal Difference Learning - Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course/en/unit2/mc-vs-td#:~:text=We%20can%20now%20update%20V,0)). Nếu sau đó từ $S'$ tác tử tiếp tục nhận thưởng hoặc bị phạt, $V(S')$ sẽ được cập nhật và ngược lại ảnh hưởng đến các lần cập nhật kế tiếp của $V(S)$ (do $V(S)$ phụ thuộc vào $V(S')$ qua TD target).

**Ưu điểm:** TD(0) có thể **học trực tuyến** trong quá trình tương tác, **không cần đợi tập kết thúc** mới cập nhật. Điều này giúp tận dụng dữ liệu hiệu quả hơn và áp dụng được cho cả các nhiệm vụ liên tục không có trạng thái kết thúc. So với MC, phương pháp TD thường có **phương sai thấp hơn** vì mỗi cập nhật chỉ dựa trên phần thưởng tức thời và giá trị ước lượng (ổn định hơn việc nhìn toàn bộ phần thưởng cuối tập) ([Lecture 4 - Model Free Techniques - MC and TD[Notes] - Omkar Ranadive](https://omkar-ranadive.github.io/posts/rl-l4-ds#:~:text=As%20the%20returns%20in%20Monte,noise%2Foutlier%20obtained%20during%20the%20episodes)). TD(0) còn có tính chất *bootstrapping* tương tự như phương pháp giải tích DP, nên trong các môi trường Markov nó có thể hội tụ nhanh về giá trị đúng (dưới các điều kiện phù hợp về $\alpha$ giảm dần, thăm đủ trạng thái...).

**Nhược điểm:** Đổi lại, phương pháp TD **bị thiên lệch (bias)** do sử dụng ước lượng $V(S_{t+1})$ thay vì giá trị thật. Sai số TD bao gồm cả sai số do ước lượng hiện tại của $V(S_{t+1})$. Tuy bias này thường giảm dần khi giá trị xấp xỉ tốt hơn, nhưng về lý thuyết MC cho ước lượng không chệch còn TD thì có chệch. Thêm nữa, TD(0) chỉ dùng thông tin một bước, có thể hội tụ chậm nếu phần thưởng có ảnh hưởng dài hạn – điều này dẫn đến ý tưởng về TD nhiều bước (TD($n$), TD($\lambda$)) để tận dụng thông tin xa hơn.

### Phương pháp TD($\lambda$) và vết lưu (Eligibility Traces)

**TD($\lambda$)** là một kỹ thuật nâng cao kết hợp các cập nhật nhiều bước, cho phép một sự **pha trộn giữa MC và TD(0)** thông qua tham số $\lambda \in [0,1]$. Có hai cách hiểu tương đương về TD($\lambda$): 

- **Forward view (quan điểm tiến):** Giá trị trạng thái được cập nhật hướng tới *$\lambda$-return* $G_t^{(\lambda)}$, tức là trung bình có trọng số của các *n-step returns* cho mọi độ dài $n$. Cụ thể, $G_t^{(n)}$ là *return* tính từ $t$ với $n$ bước đầu tiên giống MC và sau đó dùng $V$ ước lượng phần còn lại. TD($\lambda$) lấy trung bình các $G_t^{(n)}$ với trọng số $\lambda^{n-1}$ (chuẩn hóa) để thu được đích đến $G_t^{(\lambda)}$. Khi $\lambda \to 1$, $G_t^{(\lambda)}$ tiến gần tới *return* đầy đủ của MC; khi $\lambda = 0$, $G_t^{(\lambda)}$ chính là TD target một bước ([Foundations of Reinforcement Learning  Model-free RL: Monte Carlo and temporal difference (TD) learning](https://users.ece.cmu.edu/~yuejiec/ece18813B_notes/lecture8-model-free-MC-TD.pdf#:~:text=%E2%80%A2%20n%20%3D%201%3A%20TD,step%20TD%20learning)).

- **Backward view (quan điểm lùi):** Sử dụng khái niệm **vết lưu (eligibility trace)** để ghi nhớ mức độ “xứng đáng cập nhật” của mỗi trạng thái dựa trên việc nó được ghé thăm gần đây. Mỗi trạng thái $s$ có một vết lưu $E_t(s)$, ban đầu bằng 0 và được cập nhật mỗi bước: 
  $$E_t(s) = \gamma \lambda\,E_{t-1}(s) + \mathbb{I}\{S_t = s\}\,,$$ 
  trong đó $\mathbb{I}\{\cdot\}$ là hàm chỉ báo trạng thái hiện tại $S_t$ (nếu trạng thái $s$ được thăm tại thời điểm $t$ thì $E_t(s)$ tăng thêm 1). Như vậy $E_t(s)$ sẽ lớn nhất cho trạng thái vừa ghé qua, và giảm dần theo thời gian (với hệ số $\gamma\lambda$) nếu trạng thái không được thăm lại ([Foundations of Reinforcement Learning  Model-free RL: Monte Carlo and temporal difference (TD) learning](https://users.ece.cmu.edu/~yuejiec/ece18813B_notes/lecture8-model-free-MC-TD.pdf#:~:text=Eligibility%20traces%20Credit%20assignment%3A%20most,s%29%20%3D%200)). Khi nhận được **sai số TD** $\delta_t$ tại thời điểm $t$, TD($\lambda$) cập nhật *mọi trạng thái* $s$ theo vết lưu:
  $$V(s) \leftarrow V(s) + \alpha\,\delta_t\, E_t(s)\,,$$
  nghĩa là các trạng thái vừa hoặc thường xuyên ghé qua (có $E_t(s)$ cao) sẽ được điều chỉnh nhiều hơn ([Foundations of Reinforcement Learning  Model-free RL: Monte Carlo and temporal difference (TD) learning](https://users.ece.cmu.edu/~yuejiec/ece18813B_notes/lecture8-model-free-MC-TD.pdf#:~:text=%CE%B4t%20%3D%20rr%2B1%20%2B%20%CE%B3V,24)). Trường hợp đặc biệt, $\lambda = 0$ thì $E_t(s) = \mathbb{I}\{S_t=s\}$ (chỉ trạng thái hiện tại được cập nhật) và ta thu được đúng thuật toán TD(0) thông thường ([Foundations of Reinforcement Learning  Model-free RL: Monte Carlo and temporal difference (TD) learning](https://users.ece.cmu.edu/~yuejiec/ece18813B_notes/lecture8-model-free-MC-TD.pdf#:~:text=%CE%B4t%20%3D%20rr%2B1%20%2B%20%CE%B3V,24)); $\lambda = 1$ (với episode kết thúc) thì vết lưu không suy giảm, mọi trạng thái trong episode đều được cập nhật với $\delta$ cuối cùng – điều này tương đương phương pháp Monte Carlo đầu cuối.

Trong thực tế, TD($\lambda$) thường được triển khai dưới dạng backward view với vết lưu (còn gọi là **phương pháp củng cố truy hồi** – *eligibility traces*) vì tính hiệu quả tính toán. Trực giác là TD($\lambda$) cho phép trải rộng ảnh hưởng của một sai số TD đến nhiều trạng thái trước đó trong episode, thay vì chỉ trạng thái ngay lập tức như TD(0). Do đó, TD($\lambda$) thường hội tụ nhanh hơn, kết hợp được ưu điểm phương sai thấp của TD(0) và (một phần) ưu điểm không chệch của MC khi $\lambda$ gần 1. 

**Ưu điểm:** TD($\lambda$) cung cấp một **cầu nối mềm dẻo giữa TD và MC**, cho phép ta điều chỉnh tham số $\lambda$ để có hiệu quả học tốt hơn trong từng bài toán. Thông qua vết lưu, TD($\lambda$) có thể phân bổ tín dụng cho **nhiều trạng thái trong quá khứ** khi nhận được phần thưởng/phạt, giúp tăng tốc lan truyền thông tin phần thưởng xa. Nhiều nghiên cứu và thực nghiệm cho thấy việc chọn $\lambda$ trung bình (0.5~0.9) thường cho tốc độ hội tụ nhanh hơn hẳn so với chỉ TD(0) hoặc MC thuần túy.

**Nhược điểm:** Việc giới thiệu $\lambda$ làm tăng một **tham số cần điều chỉnh**. Nếu $\lambda$ không phù hợp, thuật toán có thể không đạt hiệu quả như mong muốn (ví dụ $\lambda$ quá lớn có thể gần như MC, phương sai cao; $\lambda$ quá nhỏ thì tiến triển chậm như TD(0)). Mặt khác, việc duy trì vết lưu cho tất cả trạng thái (hoặc cặp trạng thái-hành động) tốn thêm bộ nhớ và tính toán. Tuy nhiên, nhược điểm này không quá nghiêm trọng trong các bài toán có số trạng thái nhỏ hoặc khi kết hợp với kỹ thuật cắt vết (*trace decay* khi gặp trạng thái lặp).

*(Lưu ý: TD($\lambda$) có thể áp dụng tương tự cho hàm hành động $Q(s,a)$, dẫn đến các thuật toán như **SARSA($\lambda$)**, **Q-learning($\lambda$)**, nhưng trong phạm vi bài giảng, ta tập trung vào phiên bản cơ bản $\lambda=0$ của chúng.)*

## Thuật toán SARSA (TD on-policy)

**SARSA** (State-Action-Reward-State-Action) là một thuật toán TD để học **hàm giá trị hành động on-policy**, nghĩa là nó **cập nhật $Q(s,a)$ theo trải nghiệm thực tế của chính sách đang được thi hành**. Tên SARSA xuất phát từ chuỗi trải nghiệm $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$ mà thuật toán sử dụng trong bước cập nhật. Giả sử tác tử có chính sách hành động (ví dụ $\epsilon$-greedy) $\pi$ hiện tại dựa trên $Q$ đang học. Ở mỗi bước $t$, từ trạng thái $S_t$ chọn hành động $A_t$ (theo $\pi$), nhận được phần thưởng $R_{t+1}$ và chuyển đến trạng thái $S_{t+1}$. **Khác biệt của SARSA** so với TD(0) ở chỗ: thay vì cập nhật giá trị trạng thái $V(s)$, ta cập nhật trực tiếp **giá trị hành động $Q(s,a)$**, và **TD target** được lấy theo hành động $A_{t+1}$ mà *chính sách $\pi$ thực sự chọn ở trạng thái kế tiếp* (on-policy). Công thức cập nhật SARSA:

$$ 
Q(S_t, A_t) \;\leftarrow\; Q(S_t, A_t) \,+\, \alpha \,\big[\, R_{t+1} + \gamma\,Q(S_{t+1}, A_{t+1}) \,-\, Q(S_t, A_t)\,\big]\,,
$$

trong đó $A_{t+1} \sim \pi(\cdot|S_{t+1})$ được chọn theo chính sách hiện tại (ví dụ $\epsilon$-greedy) ([Reinforcement Learning Basics: Q-learning and SARSA · ADCME](https://kailaix.github.io/ADCME.jl/latest/reinforcement_learning/#:~:text=The%20SARSA%20update%20formula%20can,be%20expressed%20as)). Ta thấy TD target lúc này là $R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})$, sử dụng **hành động kế tiếp $A_{t+1}$ đúng theo chính sách đang theo**. Vì vậy SARSA là **on-policy TD control** – nó học giá trị hành động cho chính sách “hành xử” (behavior policy) và cũng chính là chính sách được đánh giá.

Sau mỗi bước cập nhật, giá trị $Q$ sẽ thay đổi, do đó chính sách $\pi$ (thường chọn hành động tham lam theo $Q$ với xác suất $1-\epsilon$) cũng dần cải thiện. SARSA thường được triển khai theo kiểu **$\epsilon$-greedy**: luôn giữ một xác suất $\epsilon$ để khám phá hành động ngẫu nhiên. Nếu $\epsilon$ được giảm dần về 0, chính sách $\pi$ sẽ hội tụ về chính sách tối ưu tham lam. Thật vậy, theo Sutton & Barto, nếu mọi cặp $(s,a)$ được thăm vô hạn lần và $\epsilon$ giảm về 0 theo thời gian, **SARSA hội tụ với xác suất 1 tới chính sách và hàm $Q$ tối ưu** ([SARSA vs Q - learning](http://tcnguyen.github.io/reinforcement_learning/sarsa_vs_q_learning.html#:~:text=,been%20published%20in%20the%20literature)).

**Ví dụ (Cliff Walking):** Xem xét bài toán đi trên vách đá (cliff) kinh điển ([SARSA vs Q - learning](http://tcnguyen.github.io/reinforcement_learning/sarsa_vs_q_learning.html#:~:text=To%20illustrate%20the%20difference%20of,explain%20it%20in%20more%20details)): một gridworld có vị trí xuất phát $S$ và mục tiêu $G$ ở góc, giữa đường có một vùng “vực” mà rơi vào đó sẽ bị phạt -100 và quay về $S$. Chính sách $\epsilon$-greedy (với $\epsilon=0.1$) ban đầu là ngẫu nhiên. Kết quả học cho thấy SARSA dần dần đánh giá cao những đường đi **an toàn** tránh xa mép vực. Chính sách tối ưu $\epsilon$-greedy mà SARSA tìm được chọn con đường vòng an toàn (dù dài bước hơn) thay vì con đường ngắn sát mép vực. Lý do: do vẫn còn xác suất $\epsilon$ để chọn hành động ngẫu nhiên, nếu đi sát vực thì khả năng rơi xuống và chịu phạt lớn là không nhỏ, làm giảm giá trị các trạng thái ven vực. SARSA “nhận thức” được rủi ro này vì **TD target của nó dùng hành động $A_{t+1}$ thực sự (bao gồm cả hành động ngẫu nhiên)** – nếu hành động đó dẫn đến rơi vực, giá trị $Q$ được điều chỉnh giảm mạnh. Kết quả, hàm $Q$ của SARSA phản ánh chi phí của việc duy trì một chính sách vẫn còn tính ngẫu nhiên, và nó ưu tiên con đường ít rủi ro hơn ([SARSA vs Q - learning](http://tcnguyen.github.io/reinforcement_learning/sarsa_vs_q_learning.html#:~:text=,Remember)). (Ngược lại, ta sẽ thấy Q-learning không tính đến rủi ro này trong cập nhật.)

**Ưu điểm:** SARSA có ưu điểm cập nhật **phù hợp với chính sách đang theo**, nên trong quá trình học, chính sách luôn “biết” về hậu quả của hành động mình sẽ thực sự thực hiện. Điều này giúp SARSA ổn định hơn trong môi trường **có khả năng nguy hiểm hoặc biến động**, bởi vì nó học cách **cân bằng giữa khám phá và khai thác** một cách trực tiếp. Như ví dụ trên, SARSA có xu hướng tìm **giải pháp an toàn hơn** trong khi vẫn còn đang khám phá. Thuật toán cũng đơn giản và đảm bảo hội tụ về tối ưu (nếu $\epsilon$ giảm dần) trong môi trường Markov theo điều kiện thông thường của phương pháp TD.

**Nhược điểm:** Do là on-policy, **hiệu quả của SARSA phụ thuộc vào chiến lược khám phá**. Nếu $\epsilon$ không được giảm đúng cách, SARSA sẽ hội tụ tới hàm $Q$ của một chính sách *$\epsilon$-soft* chứ không phải tối ưu tuyệt đối. Nói cách khác, nếu luôn giữ $\epsilon>0$, chính sách tìm được không phải tối ưu hoàn toàn mà vẫn “nhún nhường” để duy trì khám phá. Ngoài ra, so với Q-learning (sẽ bàn dưới đây), SARSA có thể **hội tụ chậm hơn** trong trường hợp không có rủi ro cao, do nó học giá trị theo hành động thực tế (bao gồm cả những hành động không phải tối ưu). Trong môi trường an toàn, điều này làm giảm tính “quyết liệt” trong việc hướng đến hành động tốt nhất. Tuy nhiên, sự khác biệt này thường nhỏ khi $\epsilon$ nhỏ, và SARSA có lợi thế về tính ổn định trong nhiều tình huống.

## Thuật toán Q-learning (TD off-policy)

**Q-learning** là thuật toán TD off-policy nổi tiếng được đề xuất bởi Watkins. Khác với SARSA, Q-learning **học hàm giá trị hành động tối ưu $Q^*(s,a)$ trực tiếp** bằng cách sử dụng *TD target theo chính sách tối ưu* thay vì theo chính sách đang thực hiện. Cụ thể, công thức cập nhật Q-learning cho mỗi bước $(S_t, A_t, R_{t+1}, S_{t+1})$ là:

$$ 
Q(S_t, A_t) \;\leftarrow\; Q(S_t, A_t) \,+\, \alpha \,\big[\, R_{t+1} + \gamma \,\max_{a'} Q(S_{t+1}, a') \,-\, Q(S_t, A_t)\,\big]\,,
$$

trong đó $\max_{a'} Q(S_{t+1}, a')$ là giá trị hành động cao nhất tại trạng thái $S_{t+1}$ theo **ước lượng hiện tại** ([Reinforcement Learning Basics: Q-learning and SARSA · ADCME](https://kailaix.github.io/ADCME.jl/latest/reinforcement_learning/#:~:text=Then%20the%20Q,can%20be%20expressed%20as)). Thành phần TD target của Q-learning: 

$$
R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a')
$$ 

chính là một mẫu của **phương trình Bellman tối ưu** đã nêu (phần bên phải, với $\max$) ([Part 1: Key Concepts in RL — Spinning Up  documentation](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#:~:text=Image%3A%20%5Cbegin%7Balign,s%27%2Ca%27%29%7D.%20%5Cend%7Balign)). Do đó, Q-learning cố gắng ước lượng $Q^*$ bất kể tác tử thực sự đang thực hiện hành động gì.

Trong quá trình học, tác tử vẫn cần khám phá (ví dụ dùng $\epsilon$-greedy dựa trên $Q$ hiện tại để chọn $A_t$), nhưng **chính sách sinh dữ liệu (behavior policy)** này có thể khác với **chính sách mục tiêu** mà Q-learning ngầm đánh giá (chính sách tối ưu greedy theo $Q$). Vì lý do này, Q-learning là phương pháp *off-policy*. Mặc dù tác tử thi hành các hành động chưa tối ưu để lấy mẫu trải nghiệm, nhưng thuật toán vẫn cập nhật $Q(s,a)$ *như thể* tác tử luôn chọn hành động tối ưu kế tiếp. 

Quay lại ví dụ **Cliff Walking**: Nếu áp dụng Q-learning, khi ở gần mép vực, TD target sẽ dùng $\max_{a'}Q(S_{t+1},a')`. Giả sử $Q$ hiện tại dự đoán hành động tối ưu là đi sát mép (để nhanh đến đích), Q-learning sẽ cập nhật $Q(S,a)$ hướng theo giả định tác tử **sẽ không rơi** (vì nó dùng giá trị lớn $\max Q$ của trạng thái kế tiếp, tương ứng hành động tối ưu nhất ở đó). Do vậy, Q-learning vẫn đánh giá cao các trạng thái ven vực nếu về dài hạn con đường đó cho phần thưởng cao, **bất chấp trong quá trình học có thể đã từng rơi vực**. Thực tế, **Q-learning tìm ra chiến lược tối ưu ngắn nhất** (men theo vực) sau khi hội tụ, nhưng trong quá trình học nó có thể nhận nhiều phần thưởng âm hơn SARSA do thử các hành động mạo hiểm (rơi vực) ([Why no falling off cliff in SARSA for the example in Sutton-Barto?](https://ai.stackexchange.com/questions/45618/why-no-falling-off-cliff-in-sarsa-for-the-example-in-sutton-barto#:~:text=Barto%3F%20ai,are%20close%20to%20the%20cliff)) ([SARSA vs Q - learning](http://tcnguyen.github.io/reinforcement_learning/sarsa_vs_q_learning.html#:~:text=,Remember)). 

Một cách khác để hiểu: Q-learning luôn **“lạc quan”** trong ước lượng giá trị – nó giả định tác tử sẽ làm tốt nhất có thể ở tương lai, nên cập nhật theo hướng đó, trong khi SARSA **“thận trọng”** hơn, cập nhật theo những gì tác tử thực sự làm (kể cả lỗi lầm do khám phá).

**Ưu điểm:** Ưu điểm lớn nhất của Q-learning là nó **hội tụ về hàm giá trị tối ưu $Q^*$** (và do đó tìm được chính sách tối ưu) ngay cả khi trong giai đoạn học tác tử theo một chính sách khác (miễn là đủ khám phá) ([SARSA vs Q - learning](http://tcnguyen.github.io/reinforcement_learning/sarsa_vs_q_learning.html#:~:text=%2A%20Q,to%20the%20optimal%20policy%20%24q)). Điều này nghĩa là Q-learning tận dụng tối đa các mẫu thu thập được để hướng đến mục tiêu dài hạn cuối cùng – phù hợp với bài toán tìm *optimal policy*. Trong thực nghiệm, Q-learning thường tìm ra chiến lược tốt nhất **nhanh hơn** so với SARSA khi môi trường không có bẫy nguy hiểm, vì nó **luôn cập nhật theo hướng tham lam nhất**. Thuật toán này cũng đơn giản và phổ biến, là nền tảng của nhiều phương pháp RL hiện đại (như Deep Q Network).

**Nhược điểm:** Do tính off-policy, Q-learning có thể gặp vấn đề trong một số trường hợp: nếu chiến lược khám phá không đủ phong phú (ví dụ $\epsilon$ quá nhỏ ngay từ đầu), Q-learning có thể đánh giá sai vì nó chưa từng thấy những hành động tồi tệ tiềm tàng. Nói cách khác, Q-learning **đòi hỏi điều kiện khám phá mạnh** (mọi $(s,a)$ phải được thử đủ nhiều) để đảm bảo hội tụ. Với function approximation (xấp xỉ hàm) thay vì bảng, Q-learning off-policy kết hợp với hàm xấp xỉ không tuyến tính thậm chí có thể không hội tụ (một vấn đề được khắc phục phần nào bởi kỹ thuật target network trong Deep Q Network). Trong môi trường có yếu tố ngẫu nhiên lớn hoặc tình huống nguy hiểm, Q-learning có thể **kém an toàn hơn** trong giai đoạn học so với SARSA, do nó không “nhìn nhận” rủi ro của việc khám phá như phân tích ở trên. Tuy nhiên, nếu ưu tiên chiến lược tối ưu cuối cùng, Q-learning vẫn là lựa chọn hàng đầu.

## Kết luận

| Phương pháp           | Đặc trưng cập nhật                     | Ưu điểm chính                            | Nhược điểm chính                                   |
|-----------------------|----------------------------------------|------------------------------------------|----------------------------------------------------|
| **Monte Carlo**       | Cập nhật từ *return* cuối tập          | Đơn giản, không cần mô hình, không chệch | Phải đợi hết tập, không áp dụng cho chuỗi không kết thúc, phương sai cao ([Foundations of Reinforcement Learning  Model-free RL: Monte Carlo and temporal difference (TD) learning](https://users.ece.cmu.edu/~yuejiec/ece18813B_notes/lecture8-model-free-MC-TD.pdf#:~:text=Caveat%20of%20Monte%20Carlo%20methods%3A,but%20consistent%20under%20mild%20conditions)) ([Lecture 4 - Model Free Techniques - MC and TD[Notes] - Omkar Ranadive](https://omkar-ranadive.github.io/posts/rl-l4-ds#:~:text=Therefore%3A)). |
| **TD(0)**             | Cập nhật mỗi bước, dùng bootstrapping  | Học online, phương sai thấp hơn, áp dụng cho tác vụ liên tục | Có độ chệch do dùng ước lượng, thông tin mỗi bước hạn chế. |
| **TD($\lambda$)**     | Kết hợp nhiều bước (vết lưu $\lambda$) | Hội tụ nhanh hơn nhờ tích hợp thông tin dài hạn, điều chỉnh bias-variance linh hoạt | Thêm tham số $\lambda$, phức tạp hơn (phải lưu vết), cần chọn $\lambda$ phù hợp. |
| **SARSA** (on-policy) | TD target theo hành động thực tế $A_{t+1}$ | Ổn định khi còn khám phá, tránh rủi ro, hội tụ về tối ưu nếu giảm $\epsilon$ ([SARSA vs Q - learning](http://tcnguyen.github.io/reinforcement_learning/sarsa_vs_q_learning.html#:~:text=,been%20published%20in%20the%20literature)) | Hội tụ về chính sách $\epsilon$-greedy (nếu $\epsilon$ cố định), có thể chậm hơn trong môi trường an toàn. |
| **Q-learning** (off-policy) | TD target tối ưu ($\max Q$)      | Hội tụ trực tiếp $Q^*$, tìm chính sách tối ưu nhanh, tận dụng mẫu hiệu quả ([SARSA vs Q - learning](http://tcnguyen.github.io/reinforcement_learning/sarsa_vs_q_learning.html#:~:text=%2A%20Q,to%20the%20optimal%20policy%20%24q)) | Cần khám phá đủ rộng, off-policy dễ mất ổn định với hàm xấp xỉ, giai đoạn học có thể thử hành động nguy hiểm. |

Nhìn chung, các phương pháp trên đều nhằm ước lượng hàm giá trị nhằm giải quyết bài toán *dự đoán* (policy evaluation) và *điều khiển* (control) trong học tăng cường. **Monte Carlo** hữu dụng khi có các episode độc lập rõ ràng và muốn tận dụng kết quả đầy đủ của mỗi episode, trong khi **TD** thích hợp cho cập nhật liên tục và thường hội tụ nhanh hơn trong các bài toán Markov. Tham số **$\lambda$** cho phép nội suy giữa hai cực trị đó, kết hợp ưu điểm của cả hai. Đối với bài toán tìm chính sách tối ưu, **SARSA** và **Q-learning** đều là những thuật toán mẫu không cần mô hình (model-free) quan trọng: SARSA phù hợp khi muốn an toàn, còn Q-learning phù hợp khi ưu tiên hiệu quả tối ưu cuối cùng. Việc lựa chọn thuật toán nào tùy thuộc vào yêu cầu cụ thể của môi trường và bài toán (yếu tố nguy cơ, tính liên tục, khả năng thăm dò...). Trong thực tế, những khái niệm và phương pháp nền tảng này còn có thể được kết hợp và mở rộng (ví dụ: **Expected SARSA**, **Double Q-learning**, **Deep Q Network**…) để đạt hiệu suất tốt hơn trên các bài toán phức tạp, nhưng tất cả đều xây dựng trên nền tảng nguyên lý của **hàm giá trị và phương trình Bellman** đã trình bày ở trên ([Monte Carlo vs Temporal Difference Learning - Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course/en/unit2/mc-vs-td#:~:text=To%20summarize%3A)) ([Part 1: Key Concepts in RL — Spinning Up  documentation](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#:~:text=The%20crucial%20difference%20between%20the,leads%20to%20the%20highest%20value)). 

