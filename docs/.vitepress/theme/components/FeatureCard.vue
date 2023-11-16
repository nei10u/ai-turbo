<template>
  <div class="feature-card" :style="backgroundStyle" @mouseover="raiseCard" @mouseleave="lowerCard" @click="goToLink">
    <div class="card-overlay">
      <h2 v-html="title"></h2>
      <p class="publishTime">{{ publishTime }}</p>
      <p class="author">{{ author }}</p>
    </div>
  </div>
</template>

<script>
export default {
  props: {
    title: String,
    publishTime: String,
    author: String,
    link: String,
    backgroundImage: String,
  },
  data() {
    return {
      raised: false,
    };
  },
  computed: {
    backgroundStyle() {
      return {
        backgroundImage: `url(${this.backgroundImage})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center center',
        transition: 'transform 0.3s ease',
        transform: this.raised ? 'scale(1.05)' : 'scale(1)',
      };
    },
  },
  methods: {
    raiseCard() {
      this.raised = true;
    },
    lowerCard() {
      this.raised = false;
    },
    goToLink() {
      window.location.href = this.link;
    },
  },
};
</script>

<style scoped>
.feature-card {
  position: relative;
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
  /* width: 300px; */
  max-width: 400px; /* 卡片最大宽度为300px */
  height: 200px;
  border-radius: 8px;
  overflow: hidden;
  cursor: pointer;
  user-select: none; /* Prevent text selection */
  -webkit-user-select: none; /* Prevent text selection for Safari */
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
  margin: 8px; /* Adjust the margin to create space around the cards */
}

.feature-card:hover {
  box-shadow: 0 12px 24px rgba(0,0,0,0.2);
}

.card-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  background: rgba(0, 0, 0, 0.5); /* Dark overlay */
  color: white;
  padding: 15px;
  text-align: center;
}

.card-overlay h2 {
  font-size: 1rem;
  margin-bottom: 0.5rem;
}

.card-overlay .publishTime {
  font-size: 0.6rem;
  margin-bottom: 0;
}

.card-overlay .author {
  font-size: 1rem;
  margin-bottom: 0;
}

/* H5 and smaller screens */
@media (max-width: 767px) {
  .feature-card {
    flex-basis: 100%; /* Each card takes full width on smaller screens */
  }
}

/* Larger than H5 screens */
@media (min-width: 768px) {
  .feature-card {
    /* Adjust size for larger screens if necessary, for example: */
    flex-basis: calc(50% - 30px); /* 2 cards per row */
  }
}

/* Even larger screens */
@media (min-width: 992px) {
  .feature-card {
    flex-basis: calc(33.333% - 30px); /* 3 cards per row */
  }
}
</style>
