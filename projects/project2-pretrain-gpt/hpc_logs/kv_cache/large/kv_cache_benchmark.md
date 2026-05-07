# KV Cache Benchmark

| Mode | Total time (s) | New tokens | Seconds/token | Tokens/sec | Peak CUDA memory (GiB) |
|---|---:|---:|---:|---:|---:|
| no_cache | 102.590 | 7680 | 0.01336 | 74.86 | 0.774 |
| kv_cache | 77.241 | 7680 | 0.01006 | 99.43 | 0.690 |

Generation settings: `max_new_tokens=512`, `temperature=0.9`, `top_k=50`, `top_p=0.95`.

## Samples: no_cache

### Prompt 1

Once upon a time, there was a little girl named Lily. She loved to paint and paint. One day, she drew a big yellow picture with bright blue and purple crayon. But then, she had a selfish idea. She wanted to draw a pretty picture for her mommy.

Lily tried to find her blue crayon, but she couldn't find it. Her mommy saw her and said, "No, Lily! Your crayon is not for looking, it's mine. You cannot find it, you won't know it."

Lily was sad and went home. She learned that it's not fair to ask for help when she needs it.Once upon a time, there was a little girl named Lily. She had a pet cat named Mittens. Mittens was very adorable and always gave her a lot of love.

One day, Lily and Mittens went for a walk in the park. They saw a shiny box on the ground. Lily and Mittens knew they were for. They opened the box and showed a beautiful red flower.

Lily was happy because she and Mittens played with the flower for a while. They had so much fun with their new toy. When it was time for dinner, Lily was sad to go back to the park, but she knew they would have more fun together.Once upon a time, there was a little girl named Lily. One day, Lily went to the park to play. She saw a pretty flower on the ground. She picked it and started to pet it. 

"Look at that!" said Lily.

"It's a pretty flower!" said her mom.

Lily giggled and said, "I wish I could catch it!"

Her mom laughed and said, "You are a brilliant girl, but you know, sometimes, someone can make us happy."Once upon a time, there was a little girl named Lily. She had a very special toy. It was her favorite toy, a shiny red car that she loved very much. Lily loved it very much and always wore it on her head.

One day, Lily went to the park to play with her friends. She saw a little boy playing with a toy car and asked him if he wanted to play with it. The boy said yes, and they played together. Lily was very happy and played with the toy car all day long.

But then, Lily's mom told her they were going to the store

### Prompt 2

Lily found a tiny red door under the old tree. She put the blue door in the trunk and closed it. It was so big that it almost touched the branches. Lily felt sad because she loved her red door. But then, something strange happened. Lily felt a little ashamed of the dark. She remembered what had happened. She went inside and told her mommy what had happened. Her mommy was worried, but then she remembered something in the woods. She hugged Lily and said, "Don't worry, we can find a good place to hide." They searched and searched, but they couldn't find anything. Lily was so happy and she hugged her mommy tightly.Once upon a time, there was a little girl named Lily. She loved to play outside and pick flowers. One day, she was going to the store with her mom. She saw a big, red basket of yellow flowers in the tree. It looked so pretty and pretty, with lots of colorful colors.

As she walked to the store, she saw a big, red ball. She wanted to get it, but it was too expensive. Lily asked her mom if she could have the ball. Her mom said no, it was too expensive and she couldn't buy it anymore.

Lily was sad, but she went home feeling sad. She didn't know that the store was too expensive, so she went home and told her mom what happened. Her mom hugged her and said, "Don't worry, I will help you fix it." Lily smiled and felt better. She was so happy and she had her favorite toy to play with.Once upon a time, there was a little girl named Lily. She loved to play in the forest and make yummy food. One day, she went to the forest and saw a big, mean dog. The dog barked and scared Lily.

Lily ran to the dog and told him that the dog was just hiding. The dog barked and Lily felt safe. She thanked the dog and said goodbye. The dog wagged his tail and licked her face, and Lily continued to play in the forest.

As she was leaving the forest, Lily remembered the dog. She ran back to her mom and said, "Look, Mommy! A puppy!" Her mom smiled and said, "That's a good dog, Lily. Thank you for trying to catch him."Once upon a time, there was a little girl named Lily. She loved to eat candy and would always ask her mommy for a candy. One

### Prompt 3

Tom wanted to help his friend learn how to share his toys. He wanted to play with Tom's cars and his cars and his car. They worked hard and made a big mess.

"Look at my car!" Tom said. He grabbed the car from Tom's hand and ran to the floor. He had an idea.

"Ben, I can help you. But you can do it. I want to play with you." Tom said. He picked up the car and took a car from his hand. It was a soft and white car.

"Here, I have this car. It's our favorite." Tom said. He gave the car to Tom.

"Thank you, Tom. You are a good friend. Can I play with your car?" Tom said.

"Sure, you can. But you have to be careful. The car is not real. It is fake. It can break the car with its mouth. You can have your own car to play with it." Tom said. He gave the car to the car.

"Thank you, Tom. You are a good friend. You are a good brother. You are a good friend. I will share and play with you." Tom said.

"Yes, Tom. I will be nice to Tom. Do you want to play with me?" Tom said.

"Yes, I want to play with you?" Tom said.

"Yes, I want to play with you. You are a good brother. You can make me happy." Tom said.

Tom and Tom agreed. They played with the car. They were having fun.

"Can we have a turn, too?" Tom asked.

"Sure, but we have to promise to share our toys with each other." Tom said.

"Okay, Tom, you are a good brother. Maybe we can share some of your toys and toys." Tom said.

They hugged and shared the toys. They were very happy. They had their favorite toys and their imagination.

"Thank you for sharing and care. You are our best friend," Tom said.

"You're welcome, Tom. You are a good friend too," Tom said.

They hugged and smiled. They were best friends.Tom and Lily were playing in the park. They liked to run and slide and slide down the slide. They had a big ball and a tag.

"Look at me, Lily! I am a princess!" Tom said

## Samples: kv_cache

### Prompt 1

Once upon a time, there was a little girl named Lily. She loved to paint and paint. One day, she drew a big yellow picture with bright blue and purple crayon. But then, she had a selfish idea. She wanted to draw a pretty picture for her mommy.

Lily tried to find her blue crayon, but she couldn't find it. Her mommy saw her and said, "No, Lily! Your crayon is not for looking, it's mine. You cannot find it, you won't know it."

Lily was sad and went home. She learned that it's not fair to ask for help when she needs it.Once upon a time, there was a little girl named Lily. She had a pet cat named Mittens. Mittens was very adorable and always gave her a lot of love.

One day, Lily and Mittens went for a walk in the park. They saw a shiny box on the ground. Lily and Mittens knew they were for. They opened the box and showed a beautiful red flower.

Lily was happy because she and Mittens played with the flower for a while. They had so much fun with their new toy. When it was time for dinner, Lily was sad to go back to the park, but she knew they would have more fun together.Once upon a time, there was a little girl named Lily. One day, Lily went to the park to play. She saw a pretty flower on the ground. She picked it and started to pet it. 

"Look at that!" said Lily.

"It's a pretty flower!" said her mom.

Lily giggled and said, "I wish I could catch it!"

Her mom laughed and said, "You are a brilliant girl, but you know, sometimes, someone can make us happy."Once upon a time, there was a little girl named Lily. She had a very special toy. It was her favorite toy, a shiny red car that she loved very much. Lily loved it very much and always wore it on her head.

One day, Lily went to the park to play with her friends. She saw a little boy playing with a toy car and asked him if he wanted to play with it. The boy said yes, and they played together. Lily was very happy and played with the toy car all day long.

But then, Lily's mom told her they were going to the store

### Prompt 2

Lily found a tiny red door under the old tree. She put the blue door in the trunk and closed it. It was so big that it almost touched the branches. Lily felt sad because she loved her red door. But then, something strange happened. Lily felt a little ashamed of the dark. She remembered what had happened. She went inside and told her mommy what had happened. Her mommy was worried, but then she remembered something in the woods. She hugged Lily and said, "Don't worry, we can find a good place to hide." They searched and searched, but they couldn't find anything. Lily was so happy and she hugged her mommy tightly.Once upon a time, there was a little girl named Lily. She loved to play outside and pick flowers. One day, she was going to the store with her mom. She saw a big, red basket of yellow flowers in the tree. It looked so pretty and pretty, with lots of colorful colors.

As she walked to the store, she saw a big, red ball. She wanted to get it, but it was too expensive. Lily asked her mom if she could have the ball. Her mom said no, it was too expensive and she couldn't buy it anymore.

Lily was sad, but she went home feeling sad. She didn't know that the store was too expensive, so she went home and told her mom what happened. Her mom hugged her and said, "Don't worry, I will help you fix it." Lily smiled and felt better. She was so happy and she had her favorite toy to play with.Once upon a time, there was a little girl named Lily. She loved to play in the forest and make yummy food. One day, she went to the forest and saw a big, mean dog. The dog barked and scared Lily.

Lily ran to the dog and told him that the dog was just hiding. The dog barked and Lily felt safe. She thanked the dog and said goodbye. The dog wagged his tail and licked her face, and Lily continued to play in the forest.

As she was leaving the forest, Lily remembered the dog. She ran back to her mom and said, "Look, Mommy! A puppy!" Her mom smiled and said, "That's a good dog, Lily. Thank you for trying to catch him."Once upon a time, there was a little girl named Lily. She loved to eat candy and would eat candy every day. One day, Lily's

### Prompt 3

Tom wanted to help his friend learn how to share his toys. He wanted to play with Tom's cars and his cars and his car. They worked hard and made a big mess.

"Look at my car!" Tom said. He grabbed the car from Tom's hand and ran to the floor. He had an idea.

"Ben, I can help you. But you can do it. I want to play with you." Tom said. He picked up the car and took a car from his hand. It was a soft and white car.

"Here, I have this car. It's our favorite." Tom said. He gave the car to Tom.

"Thank you, Tom. You are a good friend. Can I play with your car?" Tom said.

"Sure, you can. But you have to be careful. The car is not real. It is fake. It can break the car with its mouth. You can have your own car to play with it." Tom said. He gave the car to the car.

"Thank you, Tom. You are a good friend. You are a good brother. You are a good friend. I will share and play with you." Tom said.

"Yes, Tom. I will be nice to Tom. Do you want to play with me?" Tom said.

"Yes, I want to play with you?" Tom said.

"Yes, I want to play with you. You are a good brother. You can make me happy." Tom said.

Tom and Tom agreed. They played with the car. They were having fun.

"Can we have a turn, too?" Tom asked.

"Sure, but we have to promise to share our toys with each other." Tom said.

"Okay, Tom, you are a good brother. Maybe we can share some of your toys and toys." Tom said.

They hugged and shared the toys. They were very happy. They had their favorite toys and their imagination.

"Thank you for sharing and care. You are our best friend," Tom said.

"You're welcome, Tom. You are a good friend too," Tom said.

They hugged and smiled. They were best friends.Tom and Lily were playing in the park. They liked to run and slide and slide down the slide. They had a big ball and a tag.

"Look at me, Lily! I am a princess!" Tom said
