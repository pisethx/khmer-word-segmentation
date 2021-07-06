<template>
  <div>
    <header class="bg-gray-800 shadow flex flex-col">
      <div class="max-w-7xl mx-auto pt-6 px-4 sm:px-6 lg:px-8">
        <div class="flex justify-center items-center flex-col">
          <h1 class="text-3xl font-bold text-white">
            <span style="font-size: 40px; vertical-align: bottom">🇰🇭</span>
            Khmer Word Segmentation
          </h1>
          <div class="mt-3">
            <ListSelect
              label="Segmentation Method"
              :items="segmentationMethods"
              v-model="selectedMethod"
            />
          </div>
        </div>
      </div>

      <a
        class="self-end px-6 py-3"
        target="_blank"
        href="https://github.com/pisethx/khmer-word-segmentation"
      >
        <div class="flex items-center text-white">
          <CodeIcon class="h-6 w-6 mr-2" />
          Github
        </div>
      </a>
    </header>
    <main>
      <div class="grid grid-cols-1 md:grid-cols-2 py-6 sm:px-6 lg:px-24 gap-8">
        <ContentEditable v-model="originalText" />
        <ContentEditable
          :readonly="true"
          :modelValue="loading ? 'កំពុងដំណើរការ...' : segmentedText"
        />
      </div>
    </main>
  </div>
</template>

<script>
import { ref, watch } from "vue";

import ContentEditable from "./components/ContentEditable.vue";
import ListSelect from "./components/ListSelect.vue";

import { CodeIcon } from "@heroicons/vue/solid";

export default {
  components: {
    ContentEditable,
    ListSelect,
    CodeIcon,
  },
  setup() {
    const segmentationMethods = [
      { id: "ICU", name: "International Components for Unicode (ICU)" },
      { id: "SYMSPELL", name: "Symspell" },
      { id: "CRF", name: "Conditional Random Field" },
      { id: "RNN", name: "Recurrent Neural Network" },
    ];
    const selectedMethod = ref(segmentationMethods[0].id);
    const originalText = ref("សូមបញ្ចូលអត្ថបទនៅទីនេះ...");
    const segmentedText = ref("");
    const timeout = ref(null);
    const loading = ref(false);

    const segment = (text, method) => {
      return fetch(
        `http://localhost:8000/khmer-word-segmentation?text=${text}&method=${method}`
      )
        .then((res) => res.json())
        .then((res) => [res, null])
        .catch((e) => [null, e]);
    };

    const onInputChange = async () => {
      loading.value = true;

      const [res, err] = await segment(
        originalText.value,
        selectedMethod.value
      );

      loading.value = false;

      if (err) {
        segmentedText.value = "ប្រព័ន្ធដំណើរការ​មិន​ប្រក្រតី។";
        return;
      }

      const { segmented_text } = res.detail;
      segmentedText.value = segmented_text ?? "";
    };

    watch(
      () => originalText.value,
      () => {
        loading.value = true;

        if (timeout.value) clearTimeout(timeout.value);
        timeout.value = setTimeout(onInputChange, 1000);
      },
      { immediate: true }
    );

    watch(() => selectedMethod.value, onInputChange);

    return {
      selectedMethod,
      segmentationMethods,
      originalText,
      segmentedText,
      loading,
    };
  },
};
</script>

<style>
@import url("https://fonts.googleapis.com/css2?family=Kantumruy&display=swap");
@import url("https://fonts.googleapis.com/css2?family=Source+Sans+Pro&display=swap");

html {
  font-family: Kantumruy, "Source Sans Pro" !important;
}
</style>