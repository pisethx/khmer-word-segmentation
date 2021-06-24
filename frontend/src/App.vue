<template>
  <div>
    <header class="bg-gray-800 shadow">
      <div class="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
        <div class="flex justify-center items-center flex-col">
          <h1 class="text-3xl font-bold text-white">Khmer Word Segmentation</h1>
          <div class="mt-3">
            <ListSelect
              label="Segmentation Method"
              :items="segmentationMethods"
              v-model="selectedMethod"
            />
          </div>
        </div>
      </div>
    </header>
    <main>
      <div class="grid grid-cols-2 py-6 sm:px-6 lg:px-24 gap-8">
        <ContentEditable v-model="originalText" />
        <ContentEditable :readonly="true" v-model="segmentedText" />
      </div>
    </main>
  </div>
</template>

<script>
import { ref, watch } from "vue";

import ContentEditable from "./components/ContentEditable.vue";
import ListSelect from "./components/ListSelect.vue";

export default {
  components: {
    ContentEditable,
    ListSelect,
  },
  setup() {
    const segmentationMethods = [
      { id: "ICU", name: "ICU" },
      { id: "SYMSPELL", name: "Symspell" },
    ];
    const selectedMethod = ref(segmentationMethods[0].id);
    const originalText = ref("Hello");
    const segmentedText = ref("Hello");
    const timeout = ref(null);

    const segment = (text, method) => {
      return fetch(
        `http://localhost:8000/khmer-word-segmentation?text=${text}&method=${method}`
      )
        .then((res) => res.json())
        .then((res) => [res, null])
        .catch((e) => [null, e]);
    };

    const onInputChange = async () => {
      const [res, err] = await segment(
        originalText.value,
        selectedMethod.value
      );

      if (err) return;

      const { segmented_text } = res.detail;
      segmentedText.value = segmented_text;
      console.log(segmented_text.split(" "));
    };

    watch(
      () => originalText.value,
      () => {
        if (timeout) clearTimeout(timeout);
        timeout.value = setTimeout(onInputChange, 1000);
      }
    );

    watch(() => selectedMethod.value, onInputChange);

    return {
      selectedMethod,
      segmentationMethods,
      originalText,
      segmentedText,
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