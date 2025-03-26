Tuyệt vời. Mình sẽ chuẩn bị một bài giảng chi tiết bằng tiếng Việt về **Policy Gradient Methods** trong Reinforcement Learning. Bài giảng sẽ dựa trên hai tài liệu chính là *Reinforcement Learning: An Introduction* của Sutton và Barto, và *Mathematical Foundations of Reinforcement Learning* của Shiyu Zhao, cùng với các nguồn bổ sung khác nếu cần. Nội dung sẽ bao gồm:

- Giới thiệu về phương pháp Policy Gradient
- Các công thức toán học liên quan
- Giải thích chi tiết từng công thức
- Ví dụ minh họa cụ thể

Mình sẽ thông báo khi nội dung đã sẵn sàng để bạn xem.

# Phương pháp Policy Gradient trong Học tăng cường

## Mục tiêu bài giảng

- Hiểu khái niệm **Policy Gradient** trong Reinforcement Learning (Học tăng cường) và tại sao cần tối ưu trực tiếp chính sách.  
- Nắm vững các công thức toán học nền tảng: định nghĩa **chính sách** (policy), **hàm mục tiêu** $J(\theta)$, **gradient** của hàm mục tiêu, và công thức Policy Gradient cơ bản (thuật toán **REINFORCE**).  
- Giải thích được ý nghĩa của từng thành phần trong công thức Policy Gradient và cách chúng ảnh hưởng đến việc cập nhật chính sách.  
- Áp dụng kiến thức qua một ví dụ minh họa cụ thể (ví dụ: bài toán **K-armed Bandit** hoặc môi trường **CartPole**).  
- So sánh sơ bộ phương pháp Policy Gradient với các phương pháp **value-based** (dựa trên giá trị) như Q-learning để thấy rõ ưu nhược điểm của từng cách tiếp cận.

## Giới thiệu

Trong học tăng cường, hai hướng tiếp cận chính để tìm chính sách tối ưu là: (1) **dựa trên giá trị (value-based)** – ví dụ như Q-learning, SARSA – tức là học hàm giá trị rồi suy ra chính sách tối ưu; và (2) **dựa trên chính sách (policy-based)** – tức là tối ưu trực tiếp hàm mục tiêu của chính sách. **Policy Gradient** thuộc nhóm thứ hai: thay vì học một hàm giá trị rồi gián tiếp cải thiện chính sách, phương pháp này **mô hình hóa và tối ưu trực tiếp** chính sách theo hướng làm tăng phần thưởng kỳ vọng ([Policy gradients – Mastering Reinforcement Learning](https://uq.pressbooks.pub/mastering-reinforcement-learning/chapter/policy-gradients/#:~:text=As%20noted%20earlier%2C%20policy,policy%20directly%20using%20gradient%20ascent)) ([Policy gradients – Mastering Reinforcement Learning](https://uq.pressbooks.pub/mastering-reinforcement-learning/chapter/policy-gradients/#:~:text=The%20goal%20of%20a%20policy,theta%5B%2Flatex%5D%20for%20the%20particular%20MDP)). Nói cách khác, thay vì hỏi “hành động nào tốt ở trạng thái này?” thông qua $Q(s,a)$, policy gradient sẽ trực tiếp điều chỉnh xác suất chọn hành động để tối ưu **hiệu suất** của tác vụ.

Policy Gradient đặc biệt hữu ích trong các tình huống cần **chính sách ngẫu nhiên** hoặc **không gian hành động liên tục**. Ví dụ, với không gian hành động liên tục hoặc rất lớn, việc áp dụng phương pháp dựa trên giá trị gặp nhiều khó khăn (do phải ước lượng hàm giá trị cho vô hạn hành động). Trong khi đó, phương pháp policy gradient có thể **học trực tiếp tham số của phân phối** hành động (ví dụ: học các tham số $\mu$, $\sigma$ của phân phối Gaussian) ([Sutton & Barto summary chap 13 - Policy Gradient Methods | lcalem](https://lcalem.github.io/blog/2019/03/21/sutton-chap13#:~:text=13,Actions)). Nhờ đó, policy gradient trở thành nền tảng của nhiều thuật toán RL hiện đại (đặc biệt khi kết hợp với mạng nơ-ron). Tiếp theo, chúng ta sẽ đi vào các định nghĩa và công thức nền tảng của phương pháp này.

## Chính sách và hàm mục tiêu trong học tăng cường

**Chính sách (policy)** trong RL xác định cách agent chọn hành động dựa trên trạng thái. Ở đây, ta xét **chính sách ngẫu nhiên** $\pi_{\theta}(a|s)$ được tham số hóa bởi vector $\theta$. Với mỗi trạng thái $s$, $\pi_{\theta}(a|s)$ cho ta xác suất chọn hành động $a$. Chính sách có thể được biểu diễn bởi một hàm số khả vi (ví dụ: mạng nơ-ron với trọng số $\theta$). Mục tiêu của RL là **tìm được chính sách tốt nhất** (tối ưu) để tối đa hóa phần thưởng nhận được theo thời gian.

Để đánh giá “độ tốt” của một chính sách, ta định nghĩa **hàm mục tiêu** (hay hàm **hiệu suất**) $J(\theta)$. Thông thường, $J(\theta)$ được lấy là **tổng phần thưởng kỳ vọng** khi agent hành động theo chính sách $\pi_{\theta}$. Nếu $s_0$ là trạng thái bắt đầu và $G_0 = \sum_{t=0}^{T} \gamma^t r_{t+1}$ là **tổng phần thưởng thu được** (có thể dùng hệ số chiết khấu $\gamma \in [0,1]$), thì: 

$$ **J(\theta) = \mathbb{E}_{\pi_{\theta}}\!\big[\,G_0 \mid s_0\big]**, \ $$

tức là kỳ vọng của tổng phần thưởng khi xuất phát từ $s_0$ và theo chính sách $\pi_{\theta}$. (Trong trường hợp nhiều trạng thái bắt đầu, $J(\theta)$ có thể được định nghĩa là trung bình trên một phân phối trạng thái ban đầu, hoặc $J(\theta) = \sum_{s} d^\pi(s)\,V^{\pi}(s)$ với $d^\pi(s)$ là phân phối trạng thái đạt được dưới chính sách $\pi$ ([Policy Gradient Algorithms | Lil'Log](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/#:~:text=%24%24%20J%28%5Ctheta%29%20%3D%20%5Csum_,pi%28s%2C%20a%29))). Nhiệm vụ của chúng ta là tìm $\theta$ tối ưu để **tối đa hóa** $J(\theta)$.

Việc tối ưu trực tiếp $J(\theta)$ là thách thức vì $J(\theta)$ phụ thuộc phức tạp vào $\theta$: phần thưởng nhận được gián tiếp qua chuỗi các trạng thái và hành động theo $\pi_{\theta}$. Thay vì tìm cực đại của $J(\theta)$ một cách mù quáng, Policy Gradient sẽ dựa vào **gradient** (đạo hàm) của $J(\theta)$ theo $\theta$ để **leo dốc** (gradient ascent) tới cực đại. Cụ thể, ta cần tính **$\nabla_{\theta} J(\theta)$** và cập nhật $\theta$ theo hướng đó.

## Công thức Policy Gradient cơ bản (thuật toán REINFORCE)

Để tối ưu $J(\theta)$, ta áp dụng **gradient ascent** trên không gian tham số. Công thức cập nhật tham số tổng quát là: 

$$ **\theta \leftarrow \theta + \alpha \,\nabla_{\theta} J(\theta)**, $$

trong đó $\alpha$ là **tốc độ học** (learning rate) quyết định bước tiến mỗi lần cập nhật ([Policy gradients – Mastering Reinforcement Learning](https://uq.pressbooks.pub/mastering-reinforcement-learning/chapter/policy-gradients/#:~:text=If%20we%20want%20to%20follow,gradient%20and%20update%20the%20weights)). Bài toán mấu chốt là tính được **$\nabla_{\theta} J(\theta)$**, tức vector đạo hàm của $J$ theo từng tham số $\theta_i$. Định lý **Policy Gradient** (Sutton & Barto, 2018) cung cấp một công thức ước lượng gradient quan trọng: 

\[ **\nabla_{\theta} J(\theta) = \mathbb{E}_{s \sim d^{\pi},\,a \sim \pi_{\theta}} \big[\,\nabla_{\theta} \ln \pi_{\theta}(a\mid s)\;Q^{\pi}(s,a)\,\big]**, ([Policy gradients – Mastering Reinforcement Learning](https://uq.pressbooks.pub/mastering-reinforcement-learning/chapter/policy-gradients/#:~:text=%5Blatex%5Da%5B%2Flatex%5D%2C%20that%20%5Blatex%5D)) \]

trong đó $Q^{\pi}(s,a)$ là **hàm giá trị hành động** khi theo chính sách hiện tại $\pi$ (phản ánh “độ tốt” của hành động $a$ tại trạng thái $s$). Trực giác của công thức trên: ta lấy kỳ vọng (theo phân phối trạng thái-hành động dưới $\pi_{\theta}$) của **gradient log-chính-sách** $\nabla_{\theta} \ln \pi_{\theta}(a|s)$ nhân với **giá trị** $Q^{\pi}(s,a)$. Công thức này cho biết hướng thay đổi tham số $\theta$ để làm tăng xác suất những hành động có giá trị $Q$ cao và giảm xác suất những hành động $Q$ thấp. 

Trong trường hợp bài toán theo episodic (có kết thúc), ta có thể biến đổi công thức trên về dạng tổng qua từng thời điểm trong một episode. Nếu $G_t$ là **tổng phần thưởng từ thời điểm $t$ trở đi** (một ước lượng Monte Carlo của $Q^{\pi}(s_t, a_t)$), thì:

\[ **\nabla_{\theta} J(\theta) = \mathbb{E}\Big[\sum_{t=0}^{T} G_t \,\nabla_{\theta} \ln \pi_{\theta}(a_t \mid s_t)\Big]**, \]

trong đó kỳ vọng lấy theo phân phối của các episode thu được khi hành động theo $\pi_{\theta}$. Dựa trên công thức này, ta có thuật toán **REINFORCE** – một phương pháp Monte Carlo cơ bản để ước lượng và cập nhật theo policy gradient:

- Thực hiện một episode bằng cách cho agent tương tác với môi trường theo chính sách hiện tại $\pi_{\theta}$. Lưu lại chuỗi trạng thái $s_0, a_0, r_1, s_1, a_1, r_2, \ldots, s_T$ đến khi kết thúc episode.  
- Tính toán **phần thưởng tích lũy** từ mỗi thời điểm $t$: $G_t = \sum_{k=0}^{T-t-1} \gamma^k\,r_{t+k+1}$ (với $\gamma$ là hệ số chiết khấu, nếu có). Đối với bandit hoặc không chiết khấu, $G_t$ chỉ là tổng các phần thưởng sau thời điểm $t$.  
- Cập nhật tham số **theo mỗi bước thời gian** trong episode đó: 

  \[ **\theta \leftarrow \theta + \alpha\, G_t \,\nabla_{\theta} \ln \pi_{\theta}(a_t \mid s_t)**, \] 

  tức là cộng vào $\theta$ một lượng tỉ lệ với $G_t$ nhân gradient của $\ln \pi$ tại hành động đã chọn ([Policy gradients – Mastering Reinforcement Learning](https://uq.pressbooks.pub/mastering-reinforcement-learning/chapter/policy-gradients/#:~:text=,is%20the%20future%20discounted%20reward)). Mỗi cập nhật như vậy sẽ **tăng** xác suất của hành động $a_t$ nếu $G_t$ lớn (vì hướng gradient nhân với $G_t$ dương) và **giảm** xác suất nếu $G_t$ nhỏ hoặc âm. 

- Lặp lại quá trình trên (nhiều episode) cho đến khi chính sách hội tụ.

**Thuật toán REINFORCE** được xem là công thức cơ bản của policy gradient: mỗi bước cập nhật **tỉ lệ thuận với tích của phần thưởng nhận được và gradient log xác suất của hành động đã thực hiện** ([Sutton & Barto summary chap 13 - Policy Gradient Methods | lcalem](https://lcalem.github.io/blog/2019/03/21/sutton-chap13#:~:text=,that%20yields%20the%20highest%20return)). Đây chính là một **ước lượng không chệch** (unbiased estimate) của $\nabla J(\theta)$. Tuy nhiên, do sử dụng hoàn toàn mẫu Monte Carlo, phương pháp này thường có **phương sai cao**, dẫn đến học chậm. 

Một mở rộng quan trọng để giảm phương sai là sử dụng **baseline**. Định lý policy gradient tổng quát cho phép trừ đi một hàm bất kỳ $b(s)$ (không phụ thuộc vào hành động) khỏi $Q^{\pi}(s,a)$ mà **không làm đổi giá trị kỳ vọng của gradient** ([Sutton & Barto summary chap 13 - Policy Gradient Methods | lcalem](https://lcalem.github.io/blog/2019/03/21/sutton-chap13#:~:text=To%20reduce%20variance%2C%20the%20policy,s)). Cụ thể, ta có thể thay $G_t$ bằng $(G_t - b(s_t))$ trong công thức cập nhật. Lựa chọn phổ biến cho $b(s)$ là  **hàm giá trị trạng thái** $V^{\pi}(s)$ ước lượng bởi một mô hình phụ (baseline này chính là **critic** trong phương pháp actor-critic). Khi đó, $G_t - V^{\pi}(s_t)$ chính là **advantage** $A(s_t,a_t)$ – cho biết hành động $a_t$ tốt hơn hay tệ hơn mức trung bình ở trạng thái $s_t$. Sử dụng baseline sẽ làm giảm phương sai của ước lượng gradient đáng kể, giúp học nhanh hơn, trong khi **kỳ vọng của gradient không đổi** (vẫn hội tụ đến cực đại địa phương của $J(\theta)$) ([Sutton & Barto summary chap 13 - Policy Gradient Methods | lcalem](https://lcalem.github.io/blog/2019/03/21/sutton-chap13#:~:text=To%20reduce%20variance%2C%20the%20policy,s)).

Tóm lại, công thức policy gradient cơ bản (REINFORCE) cho chúng ta cách tính gradient của hàm mục tiêu và một nguyên tắc cập nhật tham số chính sách: **tăng cường những hành động dẫn đến phần thưởng cao, giảm tần suất những hành động dẫn đến phần thưởng thấp**. 

## Giải thích công thức Policy Gradient và cơ chế cập nhật

Công thức $\nabla_{\theta} J(\theta) = \mathbb{E}[\nabla_{\theta} \ln \pi_{\theta}(a|s)\,Q^{\pi}(s,a)]$ chứa đựng nhiều ý nghĩa quan trọng:

- **$\nabla_{\theta} \ln \pi_{\theta}(a|s)$**: Đây là **gradient của log xác suất** hành động, còn gọi là **score function**. Vector này cho ta biết **hướng và độ lớn** cần điều chỉnh các tham số $\theta$ để **tăng xác suất** chọn hành động $a$ tại trạng thái $s$ ([Policy gradients – Mastering Reinforcement Learning](https://uq.pressbooks.pub/mastering-reinforcement-learning/chapter/policy-gradients/#:~:text=The%20expression%20%5Blatex%5D%5Ctextrm,otherwise%20we%20decrease%20the%20probability)). Trực giác: nếu $\theta$ thay đổi một chút, $\ln \pi_{\theta}(a|s)$ sẽ thay đổi theo hướng nào? Gradient này chỉ ra hướng đó.

- **$Q^{\pi}(s,a)$**: Đây là **độ lợi (giá trị)** của hành động $a$ ở trạng thái $s$ dưới chính sách hiện tại. Nếu $Q^{\pi}(s,a)$ lớn, nghĩa là hành động $a$ tại $s$ cho kết quả tốt (phần thưởng cao); nếu $Q$ nhỏ hoặc âm, nghĩa là hành động đó không tốt trong trạng thái đó.

- **Tích $\nabla_{\theta} \ln \pi_{\theta}(a|s) \, Q^{\pi}(s,a)$**: Nếu $Q^{\pi}(s,a)$ dương và lớn, tích này sẽ **cùng hướng với $\nabla_{\theta} \ln \pi_{\theta}(a|s)$** và có độ lớn đáng kể. Điều đó đồng nghĩa với việc cập nhật $\theta$ theo hướng **tăng** $\ln \pi_{\theta}(a|s)$ (tăng xác suất chọn $a$). Ngược lại, nếu $Q^{\pi}(s,a)$ âm (hành động kém), tích sẽ đảo hướng (vì nhân số âm) – tức là **đi ngược chiều** gradient của log-policy, dẫn đến **giảm** xác suất chọn hành động $a$. Trường hợp $Q^{\pi}(s,a)$ xấp xỉ 0 (hành động trung bình), cập nhật sẽ rất nhỏ. Như vậy, mỗi cặp $(s,a)$ đóng góp vào gradient theo hướng khuyến khích hành động nếu nó tốt hơn trung bình và ngược lại nếu nó tệ hơn.

- **Kỳ vọng $\mathbb{E}_{s,a\sim \pi}[\cdot]$**: Chúng ta lấy trung bình trên tất cả các trạng thái và hành động mà chính sách có thể gặp. Trên thực tế, việc tính kỳ vọng này được thực hiện bằng cách lấy mẫu (sample) thông qua các episode. Sau đủ nhiều lần lấy mẫu, hướng trung bình của các cập nhật sẽ xấp xỉ gradient thật. Đây chính là lý do **thuật toán REINFORCE sử dụng trung bình các cập nhật từ tập các bước trong episode** để ước lượng $\nabla J(\theta)$ một cách không chệch.

- **Cập nhật tham số**: Khi thực hiện $\theta \leftarrow \theta + \alpha G_t \nabla_{\theta} \ln \pi_{\theta}(a_t|s_t)$, ta đang **đi theo hướng gradient lên dốc** của $J(\theta)$. Điều này đảm bảo rằng, với đủ số lần lặp, $\theta$ sẽ tiến tới cực đại địa phương của $J$ ([Sutton & Barto summary chap 13 - Policy Gradient Methods | lcalem](https://lcalem.github.io/blog/2019/03/21/sutton-chap13#:~:text=,that%20yields%20the%20highest%20return)). Mỗi bước cập nhật chỉ sử dụng một mẫu (episode), nên nó không hoàn hảo, nhưng **trung bình nhiều bước sẽ hội tụ về hướng đúng**. 

- **Phương sai và baseline**: Như đã đề cập, việc trừ baseline $b(s)$ chỉ ảnh hưởng đến độ lớn cập nhật (thông qua $Q^\pi(s,a) - b(s)$) chứ **không đổi dấu hay hướng** của từng cập nhật cho hành động tốt/xấu, do đó không làm sai lệch gradient kỳ vọng ([Sutton & Barto summary chap 13 - Policy Gradient Methods | lcalem](https://lcalem.github.io/blog/2019/03/21/sutton-chap13#:~:text=To%20reduce%20variance%2C%20the%20policy,s)). Điều này giúp việc học ổn định hơn. Trong thực tiễn, việc kết hợp với một ước lượng giá trị (baseline) – tạo thành thuật toán **actor-critic** – cho phép cập nhật **liên tục (online)** thay vì chờ hết episode và làm giảm phương sai cập nhật rõ rệt.

Tóm lại, công thức Policy Gradient cho thấy chiến lược cập nhật rất “tự nhiên”: **những hành động đem lại phần thưởng cao hơn mong đợi sẽ được làm cho xảy ra thường xuyên hơn, và ngược lại** ([Policy gradients – Mastering Reinforcement Learning](https://uq.pressbooks.pub/mastering-reinforcement-learning/chapter/policy-gradients/#:~:text=The%20expression%20%5Blatex%5D%5Ctextrm,otherwise%20we%20decrease%20the%20probability)). Chính sách được tinh chỉnh dần qua mỗi lần cập nhật, hướng tới tối đa hóa phần thưởng nhận được.

## Ví dụ minh họa cụ thể

Để hiểu rõ hơn cơ chế Policy Gradient, chúng ta xét hai ví dụ: một bài toán đơn giản kiểu bandit và một môi trường kinh điển trong RL.

- **Ví dụ 1: Bài toán 2-armed Bandit** – Giả sử một agent đối mặt với 2 máy đánh bạc (2 hành động khả dĩ). Chính sách $\pi_{\theta}$ có thể được định nghĩa bởi tham số $\theta$ là xác suất kéo cần **máy 1**: $\pi_{\theta}(a_1) = p = \theta$ và $\pi_{\theta}(a_2) = 1-p$. Ban đầu, $\theta$ có thể là 0.5 (chọn ngẫu nhiên mỗi máy 50-50). Mục tiêu là tìm được máy cho thưởng trung bình cao hơn và tập trung chơi máy đó.

  Áp dụng thuật toán REINFORCE cho bandit: mỗi lần chơi ta nhận được một phần thưởng $R$. Giả sử lần chơi hiện tại chọn **máy 1** và nhận thưởng $R$. Ta cập nhật tham số $\theta$ như sau:

  \[ \Delta\theta = \alpha \, R \,\nabla_{\theta} \ln \pi_{\theta}(a_1) = \alpha \, R \,\frac{\partial \ln p}{\partial \theta}. \]

  Vì $p=\theta$, $\ln \pi_{\theta}(a_1) = \ln \theta$ nên $\frac{\partial \ln \theta}{\partial \theta} = \frac{1}{\theta}$. Do đó, $\Delta\theta = \alpha R \frac{1}{\theta}$. Nếu phần thưởng $R$ nhận được là **dương lớn**, $\theta$ sẽ tăng đáng kể (tăng xác suất chọn máy 1); ngược lại nếu $R$ thấp hoặc âm, $\Delta\theta$ âm làm $\theta$ giảm (giảm xác suất chọn máy 1). Trường hợp chọn **máy 2**, ta có $\ln \pi_{\theta}(a_2) = \ln(1-\theta)$, đạo hàm $\frac{\partial \ln(1-\theta)}{\partial \theta} = -\frac{1}{1-\theta}$. Khi đó cập nhật $\Delta\theta = \alpha R (-\frac{1}{1-\theta})$. Nếu $R$ lớn, $\Delta\theta$ âm (giảm $\theta$, đồng nghĩa tăng xác suất máy 2); nếu $R$ nhỏ, $\Delta\theta$ dương (tăng $\theta$). 

  Như vậy, thuật toán policy gradient sẽ dần **dồn xác suất về máy có phần thưởng trung bình cao hơn**. Sau nhiều lần chơi, nếu máy 1 tỏ ra tốt hơn, tham số $\theta$ sẽ tiến gần 1 (chính sách chọn máy 1 gần như 100%); ngược lại nếu máy 2 tốt hơn thì $\theta$ tiến về 0. Điều này tương tự mục tiêu của các thuật toán bandit khác, nhưng ở đây ta thấy rõ cách tiếp cận gradient: **điều chỉnh xác suất theo hướng làm tăng thưởng nhận được**.

- **Ví dụ 2: Môi trường CartPole** – CartPole là bài toán kinh điển: một chiếc xe đẩy chuyển động trên đường thẳng, trên xe có một cây cột gắn bản lề. Mục tiêu của agent là **giữ cho cột không bị ngã** bằng cách di chuyển xe **sang trái hoặc phải**. Mỗi bước thời gian giữ được cột thăng bằng agent nhận phần thưởng +1, nếu cột ngã thì episode kết thúc (phần thưởng dừng lại). 

  Chính sách $\pi_{\theta}(a|s)$ ở đây có thể được tham số hóa bằng một mạng neural nhận vào **trạng thái $s$** (gồm vị trí, vận tốc của xe và góc, tốc độ góc của cột) và đầu ra là phân phối xác suất của hai hành động **trái/phải**. Ta huấn luyện chính sách này bằng policy gradient như sau: mỗi episode, agent sử dụng chính sách hiện tại để tương tác. Giả sử một episode kéo dài được $N$ bước trước khi cột ngã, tổng phần thưởng thu được $G_0 = N$ (vì mỗi bước +1). Thuật toán REINFORCE sẽ tính gradient cho **toàn bộ chuỗi hành động** trong episode đó và cập nhật $\theta$. Cụ thể, **những hành động ở đầu episode** (góp phần giữ cột lâu) nhận $G_t$ lớn sẽ có cập nhật làm **tăng** xác suất thực hiện chúng trong những lần sau; ngược lại, **những hành động cuối cùng dẫn đến ngã cột sớm** (phần thưởng sau đó bằng 0) sẽ có $G_t$ thấp, cập nhật làm **giảm** xác suất của các hành động đó trong trạng thái tương tự.

  Chẳng hạn, nếu ở một trạng thái $s$, hành động “đẩy xe sang trái” được chọn và sau đó agent trụ được rất lâu (return $G_t$ cao), thì $\theta$ sẽ được điều chỉnh để tăng $\pi_{\theta}(\text{trái}|s)$. Ngược lại, nếu hành động đó làm cột ngã ngay (return thấp), $\pi_{\theta}(\text{trái}|s)$ sẽ bị giảm. Kết quả là qua nhiều episode, chính sách $\pi_{\theta}$ sẽ **học cách cân bằng**: ở mỗi trạng thái, xác suất chọn hành động (trái/phải) sẽ tiệm cận đến giá trị tối ưu để giữ cột thăng bằng lâu nhất. Thực tế cho thấy thuật toán policy gradient (ví dụ REINFORCE hoặc các biến thể có baseline) có thể giải tốt bài toán CartPole, đạt được chính sách gần như luôn giữ được cột thăng bằng.

## So sánh Policy Gradient với phương pháp value-based (Q-learning)

Để kết thúc, hãy so sánh ngắn gọn đặc điểm của phương pháp **policy-based (Policy Gradient)** và **value-based**:

- **Cập nhật trực tiếp vs. gián tiếp**: Policy Gradient **cập nhật trực tiếp tham số của chính sách** để tăng phần thưởng kỳ vọng ([Policy gradients – Mastering Reinforcement Learning](https://uq.pressbooks.pub/mastering-reinforcement-learning/chapter/policy-gradients/#:~:text=As%20noted%20earlier%2C%20policy,policy%20directly%20using%20gradient%20ascent)). Ngược lại, Q-learning (value-based) **cập nhật hàm giá trị** ($Q(s,a)$) rồi suy ra chính sách tối ưu bằng cách chọn hành động giá trị cao nhất. Nói nôm na, policy gradient **học hành động đúng ngay**, còn value-based **học bảng điểm rồi mới chọn hành động**.

- **Chính sách ngẫu nhiên suốt quá trình học**: Trong policy gradient, ta luôn duy trì một chính sách ngẫu nhiên (và điều chỉnh phân phối xác suất). Điều này tự nhiên đảm bảo sự **khám phá** (exploration) – vì ngay cả khi chính sách gần tối ưu, nó vẫn có xác suất nhỏ chọn hành động khác để thu thập thông tin. Trong Q-learning, ta thường có chính sách $\epsilon$-greedy để cân bằng khai thác/khám phá; về cuối quá trình học, chính sách dần trở nên **quyết định** (deterministic, chọn hành động tốt nhất). Policy gradient cho phép duy trì tính ngẫu nhiên một cách linh hoạt, và thậm chí tối ưu một **chính sách ngẫu nhiên tối ưu** (nếu bài toán yêu cầu).

- **Không gian hành động liên tục**: Policy Gradient tỏa sáng khi hành động là liên tục hoặc rất nhiều. Khi đó, thay vì phải ước lượng giá trị cho từng hành động (bất khả thi), chính sách có thể sinh hành động liên tục (ví dụ xuất ra tham số của phân phối Gauss) và ta cập nhật những tham số này bằng gradient. Phương pháp value-based như Q-learning không áp dụng trực tiếp cho không gian hành động liên tục trừ khi kết hợp phương pháp xấp xỉ/học hàm ($Q$-learning phải dùng kỹ thuật đặc biệt như DDPG, về cơ bản cũng đưa về dạng policy gradient). Nói cách khác, **policy gradient cung cấp một cách tiếp cận tự nhiên cho các bài toán hành động liên tục** ([Sutton & Barto summary chap 13 - Policy Gradient Methods | lcalem](https://lcalem.github.io/blog/2019/03/21/sutton-chap13#:~:text=13,Actions)).

- **Hiệu quả mẫu và tốc độ hội tụ**: Thuật toán policy gradient (đặc biệt REINFORCE Monte Carlo) thường đòi hỏi nhiều mẫu hơn do phương sai cao – cần lấy trung bình qua nhiều episode để có hướng gradient chính xác. Ngược lại, các phương pháp value-based dùng cơ chế bootstrapping (cập nhật giá trị sử dụng một phần ước lượng của chính nó) có xu hướng **mẫu hiệu quả** hơn trong nhiều trường hợp. Tuy nhiên, bootstrapping cũng có nhược điểm: ước lượng giá trị sai lệch có thể dẫn đến **chệch** và thuật toán có thể không ổn định (như trong trường hợp Q-learning với hàm xấp xỉ phi tuyến). Policy gradient (và các biến thể có baseline) thường **ổn định hơn** và đảm bảo **hội tụ tới cực đại địa phương** của hiệu suất ([Sutton & Barto summary chap 13 - Policy Gradient Methods | lcalem](https://lcalem.github.io/blog/2019/03/21/sutton-chap13#:~:text=,high%20variance%20and%20slow%20learning)), trong khi value-based có thể phân kỳ nếu không điều chỉnh cẩn thận.

- **Kết hợp hai phương pháp**: Trên thực tế, ranh giới giữa policy-based và value-based có thể được xóa nhòa. Thuật toán **Actor-Critic** kết hợp cả hai: **actor** cập nhật chính sách theo policy gradient, còn **critic** học hàm giá trị (làm baseline) để giảm phương sai. Sự kết hợp này thừa hưởng ưu điểm của cả hai – vừa học nhanh hơn (nhờ critic) vừa áp dụng được cho không gian hành động phức tạp (nhờ actor). Nhiều thuật toán hiện đại (A2C, A3C, DDPG, PPO, v.v.) đều thuộc nhóm actor-critic, xây dựng trên nền tảng nguyên lý Policy Gradient.

**Tóm lại**, Policy Gradient cung cấp một phương pháp tiếp cận trực tiếp và **đầy sức mạnh** để tối ưu chính sách trong Reinforcement Learning. Thông qua việc cập nhật tham số theo hướng gradient của phần thưởng, ta có thể huấn luyện agent một cách **mềm dẻo** (vì chính sách luôn ngẫu nhiên trong quá trình học) và mở ra khả năng giải quyết những bài toán phức tạp mà phương pháp value-based truyền thống gặp khó khăn. Tuy phải đánh đổi bằng hiệu suất mẫu, các cải tiến như dùng baseline, kết hợp actor-critic và các kỹ thuật giảm phương sai đã khiến Policy Gradient Methods trở thành **công cụ quan trọng** trong học tăng cường hiện đại ([Policy gradients – Mastering Reinforcement Learning](https://uq.pressbooks.pub/mastering-reinforcement-learning/chapter/policy-gradients/#:~:text=The%20learning%20outcomes%20of%20this,chapter%20are)) ([Policy gradients – Mastering Reinforcement Learning](https://uq.pressbooks.pub/mastering-reinforcement-learning/chapter/policy-gradients/#:~:text=The%20goal%20of%20a%20policy,theta%5B%2Flatex%5D%20for%20the%20particular%20MDP)).

