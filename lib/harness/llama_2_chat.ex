defmodule Harness.Llama2Chat do
  @moduledoc """
  Define the Llama 2 serving.

  - https://huggingface.co/Trelis/Llama-2-7b-chat-hf-function-calling-v3
  """

  def serving() do
    # NOTE: After the model is downloaded, you can toggle to `offline: true` to
    #       only use the locally cached files and not reach out to HF at all.
    hf_token = System.fetch_env!("HF_TOKEN")
    repo = {:hf, "meta-llama/Llama-2-7b-chat-hf", auth_token: hf_token}

    {:ok, model_info} = Bumblebee.load_model(repo)

    {:ok, tokenizer} = Bumblebee.load_tokenizer(repo)

    {:ok, generation_config} = Bumblebee.load_generation_config(repo)

    generation_config =
      Bumblebee.configure(generation_config,
        max_new_tokens: 1024,
        strategy: %{type: :multinomial_sampling, top_p: 0.6}
      )

    Bumblebee.Text.generation(model_info, tokenizer, generation_config,
      compile: [batch_size: 1, sequence_length: 4096],
      stream: true,
      stream_done: true,
      defn_options: [compiler: EXLA, lazy_transfers: :never]
      # preallocate_params: true
    )
  end
end
